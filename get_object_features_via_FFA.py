from utils.inference_utils import get_features, get_features_via_batch_tensor, resize_and_pad, \
    get_weighted_FFA_features
from utils.instance_det_dataset import RealWorldDatasetWithMask, InstanceDataset
import numpy as np
import torch
from torch import nn
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
import os
import json
from tqdm import trange, tqdm
from matplotlib import colors
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from utils.instance_det_dataset import BOPDataset, SAM6DBOPDataset, OWIDDataset, MVImgDataset
import time
import math
from utils.inference_utils import FFA_preprocess, get_foreground_mask, get_cls_token

# Function to find the bounding box of the non-zero regions in the mask
def find_mask_bbox(mask_array):
    rows = np.any(mask_array, axis=1)
    cols = np.any(mask_array, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return (cmin, rmin, cmax+1, rmax+1)  # PIL uses (left, upper, right, lower)


img_size = 448
# Define transformations to be applied to the images
# used before function get_object_features_via_dataloader
# transform = transforms.Compose([
#             transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ])

# object_dataset = InstanceDataset(data_dir='./database/Objects', dataset='Object',transform=transform, imsize=img_size)
object_dataset = InstanceDataset(data_dir='./database/Objects', dataset='Object',transform=None, imsize=img_size)

# use dino v2 to extract features
encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg') #
encoder.to('cuda')
encoder.eval()

def get_FFA_feature(img_path, encoder, img_size=448):
    """used for a pair of rgb and mask images"""
    mask_path = img_path.replace('images', 'masks').replace('.jpg', '.png')
    mask = Image.open(mask_path)
    mask = mask.convert('L')

    with open(img_path, 'rb') as f:
        img = Image.open(f)
        img = img.convert('RGB')

    w, h = img.size

    if (img_size is not None) and (min(w, h) > img_size):
        img.thumbnail((img_size, img_size), Image.LANCZOS)
        mask.thumbnail((img_size, img_size), Image.BILINEAR)

        # mask.show()
    else:
        new_w = math.ceil(w / 14) * 14
        new_h = math.ceil(h / 14) * 14
        img = img.resize((new_w, new_h), Image.LANCZOS)
    # mask = mask.resize((16 , 16), Image.BILINEAR)
    img.show()
    mask.show()

    with torch.no_grad():
        preprocessed_imgs = FFA_preprocess([img], img_size).to("cuda")
        mask_size = img_size // 14
        masks = get_foreground_mask([mask], mask_size).to("cuda")
        emb = encoder.forward_features(preprocessed_imgs)

        grid = emb["x_norm_patchtokens"].view(1, mask_size, mask_size, -1)

        avg_feature = (grid * masks.permute(0, 2, 3, 1)).sum(dim=(1, 2)) / masks.sum(dim=(1, 2, 3)).unsqueeze(-1)

        return avg_feature

def get_object_masked_FFA_features(output_dir, json_filename, object_dataset, model, img_size=448):
    """get FFA features for a dataset. Mainly use this function.
    object_dataset: should have resized images and masks. No need to transform.
    """
    if os.path.exists(os.path.join(output_dir, json_filename)):
        with open(os.path.join(output_dir, json_filename), 'r') as f:
            feat_dict = json.load(f)

        object_features = torch.Tensor(feat_dict['features']).cuda()

    else:
        # Capture the start time
        start_time = time.time()
        batch_size = 32  # Define the batch size
        object_features = []

        # Initialize lists to hold a batch of images and masks
        batch_images = []
        batch_masks = []
        for i in trange(len(object_dataset)):
            img, _, mask = object_dataset[i]
            # img.show()
            mask = mask.convert('L')

            # Add the processed image and mask to the batch
            batch_images.append(img)
            batch_masks.append(mask)
            # Check if the batch is full
            if len(batch_images) == batch_size or i == len(object_dataset) - 1:
                # Process the batch
                ffa_features = get_features(batch_images, batch_masks, model, img_size=img_size)
                #ffa_features = get_cls_token(batch_images, batch_masks, model, img_size=img_size) # get class tokens

                # Extend the main feature list with the features from this batch
                object_features.append(ffa_features)

                # Clear the lists for the next batch
                batch_images = []
                batch_masks = []
        object_features = torch.cat(object_features, dim=0)

        feat_dict = dict()
        feat_dict['features'] = object_features.detach().cpu().tolist()
        end_time = time.time()

        # Calculate and print the total time
        print(f"Total running time: {end_time - start_time} seconds")

        with open(os.path.join(output_dir, json_filename), 'w') as f:
            json.dump(feat_dict, f)


    return object_features

def get_object_features_via_dataloader(output_dir, json_filename, object_dataset, model, img_size=448):
    """

    @param output_dir: save dirs
    @param json_filename: save the features
    @param object_dataset: should have transformed images and masks
    @param model: DINOv2 model
    @param img_size: 224, 336 or 448
    @return:
    """
    # json_filename = 'lmo_object_features_160.json'
    if os.path.exists(os.path.join(output_dir, json_filename)):
        with open(os.path.join(output_dir, json_filename), 'r') as f:
            feat_dict = json.load(f)

        object_features = torch.Tensor(feat_dict['features']).cuda()

    else:
        # Capture the start time
        start_time = time.time()
        batch_size = 32  # Define the batch size
        object_features = []

        # Initialize lists to hold a batch of images and masks
        dataloader = DataLoader(object_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
        for imgs, _, masks in tqdm(dataloader):
            ffa_features = get_features_via_batch_tensor(imgs, masks, model, img_size=img_size)
            object_features.append(ffa_features)

        object_features = torch.cat(object_features, dim=0)

        feat_dict = dict()
        feat_dict['features'] = object_features.detach().cpu().tolist()
        end_time = time.time()

        # Calculate and print the total time
        print(f"Total running time: {end_time - start_time} seconds")

        with open(os.path.join(output_dir, json_filename), 'w') as f:
            json.dump(feat_dict, f)


    return object_features



# demo usage:
# features = get_FFA_feature("database/Objects/099_mug_blue/images/020.jpg",  encoder, img_size=448)
# print(features.shape)

#obj_features = get_object_masked_FFA_features('./obj_FFA', 'object_features_l_reg_class.json', object_dataset, encoder, img_size=img_size)

# obj_features = get_object_features_via_dataloader('./obj_FFA', 'object_features_small.json', object_dataset, encoder, img_size=img_size)
# print(obj_features.shape)