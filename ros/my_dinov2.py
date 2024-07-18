import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.utils import make_grid, save_image
import pytorch_lightning as pl
import logging
import numpy as np
# from utils.bbox_utils import CropResizePad, CustomResizeLongestSide
from torchvision.utils import make_grid, save_image
# from utils.model_utils import BatchedData
from copy import deepcopy

descriptor_size = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitl14_reg": 1024,
    "dinov2_vitg14": 1536,
}

class BatchedData:
    """
    A structure for storing data in batched format.
    Implements basic filtering and concatenation.
    """

    def __init__(self, batch_size, data=None, **kwargs) -> None:
        self.batch_size = batch_size
        if data is not None:
            self.data = data
        else:
            self.data = []

    def __len__(self):
        assert self.batch_size is not None, "batch_size is not defined"
        return np.ceil(len(self.data) / self.batch_size).astype(int)

    def __getitem__(self, idx):
        assert self.batch_size is not None, "batch_size is not defined"
        return self.data[idx * self.batch_size : (idx + 1) * self.batch_size]

    def cat(self, data, dim=0):
        if len(self.data) == 0:
            self.data = data
        else:
            self.data = torch.cat([self.data, data], dim=dim)

    def append(self, data):
        self.data.append(data)

    def stack(self, dim=0):
        self.data = torch.stack(self.data, dim=dim)

class CropResizePad:
    def __init__(self, target_size):
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        self.target_size = target_size
        self.target_ratio = self.target_size[1] / self.target_size[0]
        self.target_h, self.target_w = target_size
        self.target_max = max(self.target_h, self.target_w)

    def __call__(self, images, boxes):
        box_sizes = boxes[:, 2:] - boxes[:, :2]
        scale_factor = self.target_max / torch.max(box_sizes, dim=-1)[0]
        processed_images = []
        for image, box, scale in zip(images, boxes, scale_factor):
            # crop and scale
            image = image[:, box[1] : box[3], box[0] : box[2]]
            image = F.interpolate(image.unsqueeze(0), scale_factor=scale.item())[0]
            # pad and resize
            original_h, original_w = image.shape[1:]
            original_ratio = original_w / original_h

            # check if the original and final aspect ratios are the same within a margin
            if self.target_ratio != original_ratio:
                padding_top = max((self.target_h - original_h) // 2, 0)
                padding_bottom = self.target_h - original_h - padding_top
                padding_left = max((self.target_w - original_w) // 2, 0)
                padding_right = self.target_w - original_w - padding_left
                image = F.pad(
                    image, (padding_left, padding_right, padding_top, padding_bottom)
                )
            assert image.shape[1] == image.shape[2], logging.info(
                f"image {image.shape} is not square after padding"
            )
            image = F.interpolate(
                image.unsqueeze(0), scale_factor=self.target_h / image.shape[1]
            )[0]
            processed_images.append(image)
        return torch.stack(processed_images)


def crop_images_and_masks(images, masks, bboxes, img_size=224):
    """
    Crop images and masks according to the given bounding boxes.

    Parameters:
        images (torch.Tensor): Tensor of shape [batch_size, channels, height, width].
        masks (torch.Tensor): Tensor of shape [batch_size, height, width].
        bboxes (torch.Tensor): Tensor of shape [batch_size, 4] with each row [y1, x1, y2, x2].

    Returns:
        cropped_images (torch.Tensor): Tensor of cropped images.
        cropped_masks (torch.Tensor): Tensor of cropped masks.
    """
    cropped_images = []
    cropped_masks = []

    for image, mask, bbox in zip(images, masks, bboxes):
        x0, y0, x1, y1 = bbox
        if x0 == x1 or y0 == y1:
            continue
        x0 = x0.int().item()
        y0 = y0.int().item()
        x1 = x1.int().item()
        y1 = y1.int().item()
        cropped_image = image[:, y0:y1, x0:x1]
        cropped_mask = mask[:, y0:y1, x0:x1]
        if cropped_image.size(1) == 0 or cropped_image.size(2) == 0:
            continue
        cropped_image = F.interpolate(cropped_image.unsqueeze(0), size=(img_size, img_size), mode='bicubic')
        cropped_mask = F.interpolate(cropped_mask.unsqueeze(0), size=(img_size, img_size), mode='bicubic')

        cropped_images.append(cropped_image)
        cropped_masks.append(cropped_mask)

    # Stack the list of tensors into a single tensor
    cropped_images = torch.cat(cropped_images, dim=0)
    cropped_masks = torch.cat(cropped_masks, dim=0)

    return cropped_images, cropped_masks

class CustomDINOv2(pl.LightningModule):
    def __init__(
        self,
        model,
        image_size=224,
        descriptor_width_size=224,
        chunk_size=32,
        patch_size=14,
    ):
        super().__init__()
        self.model = model
        self.chunk_size = chunk_size
        self.patch_size = patch_size
        self.proposal_size = image_size
        self.descriptor_width_size = descriptor_width_size
        logging.info(f"Init CustomDINOv2 done!")
        self.rgb_normalize = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        self.img_resize = T.Compose(
            [
                T.Resize((patch_size, patch_size), interpolation=T.InterpolationMode.BICUBIC)
            ])
        # use for global feature
        self.rgb_proposal_processor = CropResizePad(self.proposal_size)
        self.patch_kernel = torch.nn.AvgPool2d(kernel_size=self.patch_size, stride=self.patch_size)
        logging.info(
            f"Init CustomDINOv2 with full size={descriptor_width_size} and proposal size={self.proposal_size} done!"
        )

    def process_rgb_proposals(self, image_np, masks, boxes):
        """
        1. Normalize image with DINOv2 transfom
        2. Mask and crop each proposals
        3. Resize each proposals to predefined longest image size
        """
        num_proposals = len(masks)
        masks = masks.unsqueeze(1)
        rgb = self.rgb_normalize(image_np).to(masks.device).float()
        rgbs = rgb.unsqueeze(0).repeat(num_proposals, 1, 1, 1)
        masked_rgbs = rgbs * masks
        cropped_imgs, cropped_masks = crop_images_and_masks(masked_rgbs, masks, boxes, self.proposal_size)
        # cropped_imgs, cropped_masks = crop_images_and_masks(rgbs, masks, boxes, self.proposal_size) # if use FFA, use this line

        return cropped_imgs, cropped_masks

    @torch.no_grad()
    def compute_features(self, images, masks):
        if images.shape[0] > self.chunk_size:
            features, appearance_feat, cls_features = self.forward_by_chunk(images, masks)
        else:
            # features = self.model(images)
            emb = self.model.forward_features(images)
            grid = emb["x_norm_patchtokens"].detach()
            if len(masks.shape) == 3:
                masks = masks.unsqueeze(1)
            # masks = masks.unsqueeze(1)
            mask_size = masks.size(2) // 14
            grid = grid.view(len(images), mask_size, mask_size, -1)
            masks = F.interpolate(masks, size=(mask_size, mask_size), mode='bilinear')
            features = (grid * masks.permute(0, 2, 3, 1)).sum(dim=(1, 2)) / masks.sum(dim=(1, 2, 3)).unsqueeze(
                -1)
            appearance_feat = F.normalize((grid * masks.permute(0, 2, 3, 1)).view(len(images), mask_size*mask_size, -1), dim=-1)
            # use class token
            cls_features = emb["x_norm_clstoken"]

        return features, appearance_feat, cls_features

    @torch.no_grad()
    def forward_by_chunk(self, processed_rgbs, processed_masks):
        batch_rgbs = BatchedData(batch_size=self.chunk_size, data=processed_rgbs)
        batch_masks = BatchedData(batch_size=self.chunk_size, data=processed_masks)
        del processed_rgbs  # free memory
        del processed_masks
        features = BatchedData(batch_size=self.chunk_size)
        appearance_features = BatchedData(batch_size=self.chunk_size)
        cls_features = BatchedData(batch_size=self.chunk_size)
        for idx_batch in range(len(batch_rgbs)):
            feats, appearance_feat, cls_feats = self.compute_features(
                batch_rgbs[idx_batch], masks=batch_masks[idx_batch]
            )
            features.cat(feats)
            appearance_features.cat(appearance_feat)
            cls_features.cat(cls_feats)
        return features.data, appearance_features.data, cls_features.data

    @torch.no_grad()
    def forward_cls_token(self, image_np, proposals):
        processed_rgbs, processed_masks = self.process_rgb_proposals(
            image_np, proposals["masks"], proposals["boxes"]
        )
        return self.forward_by_chunk(processed_rgbs, processed_masks)

    @torch.no_grad()
    def forward(self, image_np, proposals):
        return self.forward_cls_token(image_np, proposals)

    @torch.no_grad()
    def forward_by_chunk_v2(self, processed_rgbs, masks):
        batch_rgbs = BatchedData(batch_size=self.chunk_size, data=processed_rgbs)
        batch_masks = BatchedData(batch_size=self.chunk_size, data=masks)
        del processed_rgbs  # free memory
        del masks
        features = BatchedData(batch_size=self.chunk_size)
        for idx_batch in range(len(batch_rgbs)):
            feats = self.compute_masked_patch_feature(
                batch_rgbs[idx_batch], batch_masks[idx_batch]
            )
            features.cat(feats)
        return features.data

    @torch.no_grad()
    def compute_masked_patch_feature(self, images, masks):
        # without preprocess
        if images.shape[0] > self.chunk_size:
            features = self.forward_by_chunk_v2(images, masks)
        else:
            features = self.model(images, is_training=True)["x_norm_patchtokens"]
            features_mask = self.patch_kernel(masks).flatten(-2) > 0.5
            # features_mask = features_mask.unsqueeze(-1).repeat(1, 1, features.shape[-1])
            features_mask = features_mask.permute(0, 2, 1)
            features = F.normalize(features * features_mask, dim=-1)
        return features