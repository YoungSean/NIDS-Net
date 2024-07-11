import os
import sys
import torch
from torchvision.transforms.functional import to_tensor
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from my_dinov2 import CustomDINOv2
from torch import nn
from absl import app, logging
from PIL import Image as PILImg
from robokit.utils import annotate, overlay_masks
from robokit.ObjDetection import GroundingDINOObjectPredictor, SegmentAnythingPredictor
from scipy.optimize import linear_sum_assignment
from src.model.utils import Detections
import cv2


class WeightAdapter(nn.Module):
    """
    Predict weights for each feature vector.
    """
    def __init__(self, c_in, reduction=4, scalar=10):
        """

        @param c_in: The channel size of the input feature vector
        @param reduction: the reduction factor for the hidden layer
        @param scalar: A scalar to scale the input feature vector
        """
        super(WeightAdapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )
        # self.ratio = ratio
        self.scalar = scalar

    def forward(self, inputs):
        inputs = self.scalar * inputs
        x = self.fc(inputs)
        x = x.sigmoid()
        x = x * inputs

        return x

def get_bounding_boxes(mask_tensor):
    bounding_boxes = []
    for mask in mask_tensor:
        # Find the indices where mask is 1 (active)
        indices = torch.nonzero(mask, as_tuple=True)
        if indices[0].size(0) == 0:  # Check if there are no active points
            # No active points in the mask, possibly skip or handle specially
            bounding_boxes.append(None)
            continue
        # Calculate min and max indices
        y_min, x_min = indices[0].min().item(), indices[1].min().item()
        y_max, x_max = indices[0].max().item(), indices[1].max().item()
        # Append bounding box coordinates (x_min, y_min, x_max, y_max)
        bounding_boxes.append((x_min, y_min, x_max, y_max))
    return bounding_boxes

def compute_similarity(obj_feats, roi_feats):
    """
    Compute Cosine similarity between object features and proposal features
    """
    roi_feats = roi_feats.unsqueeze(-2)
    sim = torch.nn.functional.cosine_similarity(roi_feats, obj_feats, dim=-1)
    return sim

def get_background_mask(foreground_masks):
    """
    Generates a background mask from a list of foreground masks.

    Args:
    foreground_masks (torch.Tensor): A tensor of shape (N, H, W) where N is the number of masks,
                                     H is the height, and W is the width of the masks.

    Returns:
    torch.Tensor: A background mask of shape (H, W).
    """
    # Normalize the masks to binary (0 and 1) where any non-zero value is considered foreground
    binary_foreground = foreground_masks > 0

    # Combine all foreground masks by taking the logical OR across all masks
    combined_foreground = torch.any(binary_foreground, dim=0)

    # Invert the mask to get the background
    background_mask = ~combined_foreground  # Using the ~ operator to invert the boolean tensor

    return background_mask

class NIDS:

    def __init__(self, template_features, use_adapter=False, adapter_path=None):
        encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        encoder.to('cuda')
        encoder.eval()
        self.encoder = encoder
        self.gdino = GroundingDINOObjectPredictor(threshold=0.15)
        self.SAM = SegmentAnythingPredictor(vit_model="vit_h")
        self.descriptor_model = CustomDINOv2(encoder)
        self.template_features = template_features
        assert self.template_features is not None, "Template features are not provided!"
        self.num_examples = self.template_features.shape[1]
        self.num_objects = self.template_features.shape[0]
        self.object_ids = [] # object ids start from 0
        self.use_adapter = use_adapter
        if self.use_adapter:
            self.adapter = WeightAdapter(1024, reduction=4).to('cuda')
            self.adapter.load_state_dict(torch.load(adapter_path))
            self.adapter.eval()


    def get_template_features(self, template_image, mask):
        """
        Get template features from the template image and mask
        Parameters
        ----------
        template_image: RGB image numpy array
        mask: a torch tensor, [H,W] with unique values for each object
        -------
        """
        # decompose the mask [H,W] to [C,H,W]. C is the number of proposals
        # Generate masks for each value from 1 to 5
        unique_values = torch.unique(mask)
        self.num_objects = len(unique_values) - 1  # -1 for the background
        self.object_ids = [i for i in range(1, self.num_objects+1)] # object ids start from 1
        masks = []
        for value in range(0, len(unique_values)):
            # the first mask is the background
            masks.append((mask == value).float())  # using .float() to convert boolean mask to float if needed

        # Stack the masks into a single tensor
        mask_tensor = torch.stack(masks, dim=0)

        # visualize the masks
        visualize = False
        if visualize:
            # Number of masks
            num_masks = mask_tensor.shape[0]

            # Set up the matplotlib figure and axes
            fig, axes = plt.subplots(1, num_masks, figsize=(15, 5))  # Adjust figsize to your needs

            # Plot each mask
            for i in range(num_masks):
                ax = axes[i] if num_masks > 1 else axes  # Handle the case of a single subplot
                ax.imshow(mask_tensor[i].cpu(), cmap='gray')  # Use gray scale to visualize the mask
                ax.axis('off')  # Turn off axis
                ax.set_title(f'Mask {i + 1}')  # Title with mask number

            # Display all the plots
            plt.show()

        fg_mask = mask_tensor[1:]  # exclude the background mask
        bounding_boxes = get_bounding_boxes(fg_mask)
        # Convert bounding boxes to a tensor
        bounding_box_tensor = torch.tensor(bounding_boxes)
        proposals = dict()
        proposals["masks"] = fg_mask.to(
            torch.float32)  # to N x H x W, torch.float32 type as the output of fastSAM
        proposals["boxes"] = bounding_box_tensor

        query_FFA_decriptors, query_appe_descriptors, query_cls_descriptors = self.descriptor_model(template_image,
                                                                                               proposals)
        self.template_features = nn.functional.normalize(query_FFA_decriptors, dim=1, p=2)
        return mask_tensor


    def step(self, image_np):
        image_pil = Image.fromarray(image_np).convert("RGB")
        # image_pil.show()
        bboxes, phrases, gdino_conf = self.gdino.predict(image_pil, "objects")
        w, h = image_pil.size  # Get image width and height
        # Scale bounding boxes to match the original image size
        image_pil_bboxes = self.gdino.bbox_to_scaled_xyxy(bboxes, w, h)
        image_pil_bboxes, masks = self.SAM.predict(image_pil, image_pil_bboxes)
        proposals = dict()
        proposals["masks"] = masks.squeeze(1).to(
            torch.float32)  # to N x H x W, torch.float32 type as the output of fastSAM
        proposals["boxes"] = image_pil_bboxes
        query_FFA_decriptors, query_appe_descriptors, query_cls_descriptors = self.descriptor_model(image_np, proposals)
        query_decriptors = query_FFA_decriptors
        if self.use_adapter:
            with torch.no_grad():
                query_decriptors = self.adapter(query_decriptors)
        scene_feature = nn.functional.normalize(query_decriptors, dim=1, p=2)
        template_features = self.template_features.view(self.num_objects * self.num_examples, -1)
        num_example = self.num_examples
        num_object = self.num_objects
        sim_mat = compute_similarity(template_features, scene_feature)
        sim_mat = sim_mat.view(len(scene_feature), num_object, num_example)
        sims, _ = torch.max(sim_mat, dim=2)  # choose max score over profile examples of each object instance
        max_ins_sim, initial_result = torch.max(sims, dim=1)

        # build new segment info and masks!
        # for mask, we need to add the background mask back since some foreground masks are removed
        # then save these results to the result_saver
        # Convert to costs
        # Replace NaN values with zero
        sims[torch.isnan(sims)] = 0
        max_value = sims.max().item() + 1
        cost_matrix = max_value - sims.cpu().numpy()

        # Applying Hungarian algorithm to find minimum cost matches
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Filtering out matches below a certain similarity threshold
        threshold = 0.5  # Define your threshold here
        matches = [(row, col) for row, col in zip(row_ind, col_ind) if sims[row, col] >= threshold]
        new_fg_mask = []
        for row, col in matches:
            new_fg_mask.append(masks[row] * (col + 1))
        if len(new_fg_mask) == 0:
            return torch.zeros([1, image_np.shape[0], image_np.shape[1]], device='cuda')
        new_mask = torch.stack(new_fg_mask, dim=0)
        new_mask = new_mask.squeeze(1) # remove the channel dimension -> [N, H ,W]
        visualize = False
        if visualize:
            # Number of masks
            num_masks = new_mask.shape[0]

            # Set up the matplotlib figure and axes
            fig, axes = plt.subplots(1, num_masks, figsize=(15, 5))  # Adjust figsize to your needs

            # Plot each mask
            for i in range(num_masks):
                ax = axes[i] if num_masks > 1 else axes  # Handle the case of a single subplot
                ax.imshow(new_mask[i].cpu(), cmap='gray')  # Use gray scale to visualize the mask
                ax.axis('off')  # Turn off axis
                ax.set_title(f'Mask {i + 1}')  # Title with mask number

            # Display all the plots
            plt.show()

        foreground_masks = new_mask > 0.5
        background_mask = get_background_mask(foreground_masks)
        new_mask = torch.cat([background_mask.unsqueeze(0), new_mask], dim=0)

        num_masks = new_mask.shape[0]

        # combine these masks to a single mask
        mask = torch.zeros([new_mask.shape[-2], new_mask.shape[-1]], device=new_mask.device)
        for i in range(num_masks):
            mask = torch.max(mask, new_mask[i])
        unique_values = torch.unique(mask)
        # print(unique_values)
        visualize = True
        if visualize:
            # Number of masks
            num_masks = 1

            # Set up the matplotlib figure and axes
            fig, axes = plt.subplots(1, num_masks, figsize=(15, 5))  # Adjust figsize to your needs

            # Plot each mask
            for i in range(num_masks):
                ax = axes[i] if num_masks > 1 else axes  # Handle the case of a single subplot
                ax.imshow(mask.cpu())  # Use gray scale to visualize the mask
                ax.axis('off')  # Turn off axis
                ax.set_title(f'Mask {i + 1}')  # Title with mask number

            # Display all the plots
            plt.show()

        return new_mask
