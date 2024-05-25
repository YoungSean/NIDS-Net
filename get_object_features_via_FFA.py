from utils.inference_utils import get_FeatUp_features, get_features, get_features_via_batch_tensor, resize_and_pad, \
    get_weighted_FFA_features
from utils.inference_utils import compute_similarity
from utils.instance_det_dataset import RealWorldDatasetWithMask, FewSOLRealObjects, InstanceDataset
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
from weighted_FFA import FeatureMapWeightModel
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


# object_dataset = RealWorldDatasetWithMask('database/Objects', "Object", transform=None, imsize=448)
# LMO_dataset = BOPDataset(data_dir='./datasets/lmo/test_video', transform=None, imsize=448, freq=1)
# YCBV_dataset = BOPDataset(data_dir='./datasets/ycbv/test_video', transform=None, imsize=448)

# print(len(SAM6D_LMO_dataset))

img_size = 448
# Define transformations to be applied to the images
transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

# RoboTools_dataset = BOPDataset(data_dir='./datasets/RoboTools/test_video', transform=transform, imsize=448, freq=4)
# print(len(RoboTools_dataset))
# x = RoboTools_dataset[0]

# print(len(OWID_dataset))
    # Create an instance of your custom dataset
# fewsol_dataset = FewSOLRealObjects(data_dir='./FewSOL/data/real_objects', transform=None, imsize=448)
# synthetic_fewsol_dataset = FewSOLRealObjects(data_dir='./FewSOL/data/synthetic_objects', transform=None, imsize=448)
# object_dataset = InstanceDataset(data_dir='./database/Objects', dataset='Object',transform=transform, imsize=img_size)
object_dataset = InstanceDataset(data_dir='./database/Objects', dataset='Object',transform=None, imsize=img_size)
# mvimg_dataset = MVImgDataset(data_dir='./datasets/MVImgNet/data', transform=transform, imsize=img_size)
# LMO_SAM6D_dataset = SAM6DBOPDataset(data_dir='./datasets/SAM6D_BOP/lmo', transform=transform, imsize=448, freq=1)


# mvimg_dataset = MVImgDataset(data_dir='./datasets/MVImgNet/data', transform=transform, imsize=img_size)

# use dino v2 to extract features
encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg') #
encoder.to('cuda')
encoder.eval()

import matplotlib.pyplot as plt
from featup.util import pca, remove_axes
from pytorch_lightning import seed_everything
import torch
import torch.nn.functional as F


@torch.no_grad()
def plot_feats(image, lr):
    assert len(image.shape) == len(lr.shape) == 3
    seed_everything(0)
    lr_feats_pca, _ = pca([lr.unsqueeze(0)])
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(image.permute(1, 2, 0).detach().cpu())
    ax[0].set_title("Image")
    plt.imshow(lr_feats_pca[0].permute(1, 2, 0).detach().cpu())

    remove_axes(ax)
    plt.show()

def get_FFA_feature(img_path, encoder, img_size=448):
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
        lr_feats_pca, _ = pca([grid.permute(0,3,1,2)])

        plt.imshow(lr_feats_pca[0][0].permute(1, 2, 0).detach().cpu())
        plt.axis('off')
        plt.savefig("test1.png", bbox_inches='tight')
        plt.show()

        masked_feat = (grid * masks.permute(0, 2, 3, 1))
        hr_feats_pca, _ = pca([masked_feat.permute(0, 3, 1, 2)])
        plt.imshow(hr_feats_pca[0][0].permute(1, 2, 0).detach().cpu())
        plt.axis('off')
        plt.savefig("test2.png", bbox_inches='tight')
        plt.show()

        avg_feature = (grid * masks.permute(0, 2, 3, 1)).sum(dim=(1, 2)) / masks.sum(dim=(1, 2, 3)).unsqueeze(-1)

        return avg_feature


def get_object_masked_FFA_features(output_dir, json_filename, object_dataset, model, img_size=448, crop_img=False):
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
            mask_array = np.array(mask)
            if crop_img:
                # Convert the mask to binary format (assuming mask values should be 1, not 255)
                mask_array = (mask_array > 0).astype(np.uint8)  # Now mask is binary (1 for mask, 0 for background)

                # Find the bounding box of the masked area
                bbox = find_mask_bbox(mask_array)

                # Crop the image and the mask using the bounding box
                cropped_image = img.crop(bbox)
                cropped_mask = mask.crop(bbox)

                img = resize_and_pad(cropped_image, final_size=(img_size, img_size),
                                     padding_color_image=(255, 255, 255))

                # Convert the mask to grayscale, resize, and pad it
                mask = resize_and_pad(cropped_mask, final_size=(img_size, img_size), padding_color_mask=0)
            # ffa_feature = get_features([img], [mask], model, img_size=img_size)
            # object_features.append(ffa_feature)
            # Add the processed image and mask to the batch
            batch_images.append(img)
            batch_masks.append(mask)
            # Check if the batch is full
            if len(batch_images) == batch_size or i == len(object_dataset) - 1:
                # Process the batch
                # ffa_features = get_features(batch_images, batch_masks, model, img_size=img_size)
                ffa_features = get_cls_token(batch_images, batch_masks, model, img_size=img_size) # get class tokens

                # Extend the main feature list with the features from this batch
                object_features.append(ffa_features)

                # Clear the lists for the next batch
                batch_images = []
                batch_masks = []
        object_features = torch.cat(object_features, dim=0)
        # ffm_features = torch.cat(ffm_features, dim=0)

        feat_dict = dict()
        feat_dict['features'] = object_features.detach().cpu().tolist()
        # feat_dict['ffm_features'] = ffm_features.detach().cpu().tolist()
        # Capture the end time
        end_time = time.time()

        # Calculate and print the total time
        print(f"Total running time: {end_time - start_time} seconds")

        with open(os.path.join(output_dir, json_filename), 'w') as f:
            json.dump(feat_dict, f)

    # normalize features
    # object_features = nn.functional.normalize(object_features, dim=1, p=2)

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

    # normalize features
    # object_features = nn.functional.normalize(object_features, dim=1, p=2)

    return object_features
def get_object_weighted_masked_FFA_features(output_dir, object_dataset, model, weighted_cnn, img_size=448, crop_img=False):
    json_filename = 'weighted_relu_FFA_object_features_vitl_reg_epoch20_res.json'
    if os.path.exists(os.path.join(output_dir, json_filename)):
        with open(os.path.join(output_dir, json_filename), 'r') as f:
            feat_dict = json.load(f)

        object_features = torch.Tensor(feat_dict['features']).cuda()

    else:
        batch_size = 8  # Define the batch size
        object_features = []

        # Initialize lists to hold a batch of images and masks
        batch_images = []
        batch_masks = []
        for i in trange(len(object_dataset)):
            img, _, mask = object_dataset[i]
            # img.show()
            mask = mask.convert('L')
            mask_array = np.array(mask)
            if crop_img:
                # Convert the mask to binary format (assuming mask values should be 1, not 255)
                mask_array = (mask_array > 0).astype(np.uint8)  # Now mask is binary (1 for mask, 0 for background)

                # Find the bounding box of the masked area
                bbox = find_mask_bbox(mask_array)

                # Crop the image and the mask using the bounding box
                cropped_image = img.crop(bbox)
                cropped_mask = mask.crop(bbox)

                img = resize_and_pad(cropped_image, final_size=(img_size, img_size),
                                     padding_color_image=(255, 255, 255))

                # Convert the mask to grayscale, resize, and pad it
                mask = resize_and_pad(cropped_mask, final_size=(img_size, img_size), padding_color_mask=0)
            # ffa_feature = get_weighted_FFA_features([img], [mask], model, weighted_cnn, img_size=img_size)
            # object_features.append(ffa_feature)
            # Add the processed image and mask to the batch
            batch_images.append(img)
            batch_masks.append(mask)
            # Check if the batch is full
            if len(batch_images) == batch_size or i == len(object_dataset) - 1:
                # Process the batch
                ffa_features = get_weighted_FFA_features(batch_images, batch_masks, model, weighted_cnn,
                                                         img_size=img_size)

                # Extend the main feature list with the features from this batch
                object_features.append(ffa_features)

                # Clear the lists for the next batch
                batch_images = []
                batch_masks = []
        object_features = torch.cat(object_features, dim=0)
        # ffm_features = torch.cat(ffm_features, dim=0)

        feat_dict = dict()
        feat_dict['features'] = object_features.detach().cpu().tolist()
        # feat_dict['ffm_features'] = ffm_features.detach().cpu().tolist()

        with open(os.path.join(output_dir, json_filename), 'w') as f:
            json.dump(feat_dict, f)

    # normalize features
    # object_features = nn.functional.normalize(object_features, dim=1, p=2)

    return object_features

def get_tokens(encoder, dataset):
    output_dir = './obj_tokens'

    for i in trange(len(dataset)):
        img, mask, label = dataset[i]
        img = img.unsqueeze(0)
        mask_size = mask.shape[-1]
        emb = encoder.forward_features(img.cuda())
        tokens = emb["x_norm_patchtokens"].view(len(img), mask_size, mask_size, -1)
        # object_tokens = torch.cat(object_tokens, dim=0)
        # object_masks = torch.cat(object_masks, dim=0)
        feat_dict = dict()
        feat_dict['token'] = tokens[0].detach().cpu().tolist()
        feat_dict['mask'] = mask.detach().cpu().tolist()
        with open(os.path.join(output_dir, 'insDet_object_tokens_vitl_reg',f'{str(i).zfill(6)}.json'), 'w') as f:
            json.dump(feat_dict, f)



# object_features = get_object_masked_FFA_features('./obj_FFA', 'fewsol_synthetic_object_features_vitl14_reg.json',synthetic_fewsol_dataset, encoder, img_size=img_size, crop_img=False)
# object_features = get_object_FeatUp_FFA_features('./obj_FFA', object_dataset, encoder)


# object_features = get_object_weighted_masked_FFA_features('./obj_weighted_FFA', object_dataset, encoder, weighted_cnn, img_size=img_size, crop_img=False)
# print(object_features.shape)
# ycbv_features = get_object_masked_FFA_features('./BOP_obj_feat', YCBV_dataset, encoder, img_size=img_size, crop_img=False)
# print(ycbv_features.shape)

# robo_features = get_object_features_via_dataloader('./RoboTools_obj_feat', 'object_features3.json',RoboTools_dataset, encoder, img_size=img_size)
# print(robo_features.shape)



# get_FFA_feature("database/Objects/099_mug_blue/images/020.jpg",  encoder, img_size=448)

# obj_features = get_object_masked_FFA_features('./obj_FFA', 'object_features_l_reg_class.json', object_dataset, encoder, img_size=img_size)

# obj_features = get_object_features_via_dataloader('./obj_FFA', 'object_features_small.json', object_dataset, encoder, img_size=img_size)
# print(obj_features.shape)