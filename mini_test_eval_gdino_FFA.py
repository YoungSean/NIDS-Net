#!/usr/bin/env python
# coding: utf-8

# Testing:
# 1. Object Proposal Generation by Grounded SAM
# 2. Feature Extraction for proposals by DINOv2
# 3. Proposal/Instance Matching -- Stable Matching

# In[6]:
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import argparse
import glob
import json
import logging
import math
import os
import sys
import random
import re
import numpy as np
from typing import List, Optional
from PIL import Image, ImageFile

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torchvision import transforms as pth_transforms
from tqdm import tqdm, trange
import time
import torchvision.transforms as T

sys.path.append("../detectron2")
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances


sys.path.append("../dinov2-main")


sys.path.append(".")
from utils.visualizer import ColorMode, Visualizer
from utils.instance_det_dataset import RealWorldDataset
from utils.inference_utils import compute_similarity, stableMatching, get_bbox_masks_from_gdino_sam, \
    get_object_proposal, getColor, create_instances, nms, apply_nms, get_features
from adapter import ModifiedClipAdapter, WeightAdapter


logger = logging.getLogger("dinov2")

def get_args_parser(
        description: Optional[str] = None,
        parents: Optional[List[argparse.ArgumentParser]] = [],
        add_help: bool = True,
):

    parents = []

    parser = argparse.ArgumentParser(
        description=description,
        parents=parents,
        add_help=add_help,
    )
    parser.add_argument(
        "--train_path",
        default="../database_mini/train",
        type=str,
        help="Path to train dataset.",
    )
    parser.add_argument(
        "--test_path",
        default="../database_mini/test",
        type=str,
        help="Path to test dataset.",
    )
    parser.add_argument(
        "--imsize",
        default=224,
        type=int,
        help="Image size",
    )

    parser.add_argument(
        "--output_dir",
        default="./output",
        type=str,
        help="Path to save outputs.")
    parser.add_argument("--num_workers", default=0, type=int, help="Number of data loading workers per GPU.")

    parser.add_argument(
        "--gather-on-cpu",
        action="store_true",
        help="Whether to gather the train features on cpu, slower"
             "but useful to avoid OOM for large datasets (e.g. ImageNet22k).",
    )

    parser.set_defaults(
        train_dataset="Object",
        test_dataset="Scene",
        batch_size=1,
        num_workers=0,
    )
    return parser


# In[8]:
# set the scene level here
scene_level = 'hard'  # all / easy / hard
# Default args and initialize model
args_parser = get_args_parser(description="Grounded SAM-DINOv2 Instance Detection")
imsize = 448
tag = "mask"  # bbox
args = args_parser.parse_args(args=[
                                    "--train_path", "database/Objects",
                                    "--test_path", "database/Data/test_1_"+scene_level,  # test_002
                                    "--output_dir", "exps/eval_ffa_"+scene_level+"4_gdino0.15t_vitl14_reg_" + str(imsize) + "_" + tag,
                                    ])
os.makedirs(args.output_dir, exist_ok=True)

encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
encoder.to('cuda')
encoder.eval()

use_adapter = True
adapter_type = "weight"
if use_adapter:
    input_features = 1024 #768, 1024, the vector dimension
    if adapter_type == "clip":
        # adapter_args = 'Ins_clip_ratio_0.6_temp_0.05_epoch_40_lr_0.0001_bs_1024_vec_reduction_4_L2e4_vitl_reg'
        adapter_args = 'Ins_0413__ratio_0.6_temp_0.05_epoch_40_lr_0.0001_bs_512_vec_reduction_4_L2e4_vitl_reg'
        model_path = 'adapter_weights/adapter2FC/' + adapter_args + '_weights.pth'
        adapter = ModifiedClipAdapter(input_features, reduction=4, ratio=0.6).to('cuda')
    elif adapter_type == "weight":
        adapter_args = 'Ins_weighted_10sigmoid_ratio_0.6_temp_0.05_epoch_40_lr_0.001_bs_1024_vec_reduction_4_L2e4_vitl_reg'
        model_path = 'adapter_weights/adapter2FC/' + adapter_args + '_weights.pth'
        adapter = WeightAdapter(input_features, reduction=4).to('cuda')


    # Load the weights
    adapter.load_state_dict(torch.load(model_path))

    # If you plan to only evaluate the model, switch to eval mode
    adapter.eval()

    print('Model weights loaded and model is set to evaluation mode.')

# ### Traning (Feature Extraction on Object Instance Profile Images by DINOv2)

# In[9]:


output_dir = './obj_FFA'
json_filename = 'object_features_vitl14_reg.json'
if use_adapter:
    output_dir = './adapted_obj_feats'
    json_filename = adapter_args+'.json'

with open(os.path.join(output_dir, json_filename), 'r') as f:
    feat_dict = json.load(f)
print(f"Loaded {os.path.join(output_dir, json_filename)}")
object_features = torch.Tensor(feat_dict['features']).cuda()
object_features = nn.functional.normalize(object_features, dim=1, p=2)

do_matching = True


# In[9]:

transform = pth_transforms.Compose([pth_transforms.ToTensor(),])
object_dataset = RealWorldDataset(args.train_path, args.train_dataset, transform=transform, imsize=args.imsize)


# In[10]:
from absl import app, logging
from PIL import Image as PILImg

from robokit.ObjDetection import GroundingDINOObjectPredictor, SegmentAnythingPredictor

logging.info("Initialize object detectors")
gdino = GroundingDINOObjectPredictor(use_vitb=False, threshold=0.15)
SAM = SegmentAnythingPredictor(vit_model="vit_h")


# In[11]:

image_dir = []
proposals_list = []
scene_name_list = []
# source_list = sorted(glob.glob(os.path.join(args.test_path, '*')))
transform = pth_transforms.Compose([pth_transforms.ToTensor(),])
scene_features_list = []
source_dir = os.path.join(args.test_path, 'images')


image_paths = sorted([p for p in glob.glob(os.path.join(source_dir, '*'))
                      if re.search('/*\.(jpg|jpeg|png|gif|bmp|pbm)', str(p))])
image_dir.extend(image_paths)

start_time = time.time()
for image_path in tqdm(image_paths):
    image_pil = PILImg.open(image_path).convert("RGB")
    scene_name = os.path.basename(image_path).split('.')[0]
    scene_name_list.append(scene_name)
    accurate_bboxs, masks = get_bbox_masks_from_gdino_sam(image_path, gdino, SAM, visualize=False)
    mask = masks.cpu().numpy()
    accurate_bboxs = accurate_bboxs.cpu().numpy()
    # set ratio = 0.25 since we are evaluating on the ground truth results of images with 1/4 resolution. Only set for the high-resolution dataset
    rois, sel_rois, cropped_imgs, cropped_masks = get_object_proposal(image_path, accurate_bboxs, masks, tag=tag, ratio=0.25, save_rois=False, output_dir=args.output_dir)
    scene_features = []
    num_imgs = len(cropped_imgs)
    # for i in range(0, num_imgs, 16):
    for i in range(num_imgs):
        img = cropped_imgs[i]
        mask = cropped_masks[i]
        ffa_feature = get_features([img], [mask], encoder, img_size=imsize)
        # img = cropped_imgs[i:i+32]
        # mask = cropped_masks[i:i+32]
        # ffa_feature = get_features(img, mask, encoder, img_size=imsize)
        with torch.no_grad():
            if use_adapter:
                ffa_feature = adapter(ffa_feature)
        scene_features.append(ffa_feature)
    scene_features = torch.cat(scene_features, dim=0)
    scene_features = nn.functional.normalize(scene_features, dim=1, p=2)

    scene_features_list.append(scene_features)
    # total_proposals[scene_name] = sel_rois
    proposals_list.append(sel_rois)



# In[7]:

num_object = 100 # number of instances in the dataset
num_example = len(object_features) // num_object

score_thresh_predefined = 0.6

results = []
    
for idx, scene_feature in enumerate(scene_features_list):
    sim_mat = compute_similarity(object_features, scene_feature)
    sim_mat = sim_mat.view(len(scene_feature), num_object, num_example)
    sims, _ = torch.max(sim_mat, dim=2)  # choose max score over profile examples of each object instance

    max_ins_sim, initial_result = torch.max(sims, dim=1)

    proposals = proposals_list[idx]
    num_proposals = len(proposals)

    ########################################## Stable Matching Strategy ##########################################

    if do_matching:
        # ------------ ranking and sorting ------------
        # Initialization
        sel_obj_ids = [str(v) for v in list(np.arange(num_object))]  # ids for selected obj
        sel_roi_ids = [str(v) for v in list(np.arange(len(scene_feature)))]  # ids for selected roi

        # Padding
        max_len = max(len(sel_roi_ids), len(sel_obj_ids))
        sel_sims_symmetric = torch.ones((max_len, max_len)) * -1
        sel_sims_symmetric[:len(sel_roi_ids), :len(sel_obj_ids)] = sims.clone()

        pad_len = abs(len(sel_roi_ids) - len(sel_obj_ids))
        if len(sel_roi_ids) > len(sel_obj_ids):
            pad_obj_ids = [str(i) for i in range(num_object, num_object + pad_len)]
            sel_obj_ids += pad_obj_ids
        elif len(sel_roi_ids) < len(sel_obj_ids):
            pad_roi_ids = [str(i) for i in range(len(sel_roi_ids), len(sel_roi_ids) + pad_len)]
            sel_roi_ids += pad_roi_ids

        # ------------ stable matching ------------
        matchedMat = stableMatching(
            sel_sims_symmetric.detach().data.cpu().numpy())  # predMat is raw predMat
        predMat_row = np.zeros_like(
            sel_sims_symmetric.detach().data.cpu().numpy())  # predMat_row is the result after stable matching
        Matches = dict()
        for i in range(matchedMat.shape[0]):
            tmp = matchedMat[i, :]
            a = tmp.argmax()
            predMat_row[i, a] = tmp[a]
            Matches[sel_roi_ids[i]] = sel_obj_ids[int(a)]
        # print("Done!")

        # ------------ thresholding ------------
        preds = Matches.copy()
        # for key, value in Matches.items():
        #     if sel_sims_symmetric[int(sel_roi_ids.index(key)), int(sel_obj_ids.index(value))] <= score_thresh_predefined:
        #         del preds[key]
        #         continue
        
        # ------------ save per scene results ------------

        for k, v in preds.items():
            if int(k) >= num_proposals:
                break
            # if float(sims[int(k), int(v)]) < score_thresh_predefined:
            #     continue
            result = dict()
            result['image_id'] = proposals[int(k)]['image_id']
            result['category_id'] = int(v)
            result['bbox'] = proposals[int(k)]['bbox']
            result['score'] = float(sims[int(k), int(v)])
            result['image_width'] = proposals[int(k)]['image_width']
            result['image_height'] = proposals[int(k)]['image_height']
            result['scale'] = proposals[int(k)]['scale']
            results.append(result)
    else:
        THRESHOLD_OBJECT_SCORE = 0.4
        for i in range(num_proposals):
            if float(max_ins_sim[i]) < THRESHOLD_OBJECT_SCORE:
                continue
            result = dict()
            result['image_id'] = proposals[i]['image_id']
            result['category_id'] = initial_result[i].item()
            result['bbox'] = proposals[i]['bbox']
            result['score'] = float(max_ins_sim[i])
            result['image_width'] = proposals[i]['image_width']
            result['image_height'] = proposals[i]['image_height']
            result['scale'] = proposals[i]['scale']
            results.append(result)

# Capture the end time
end_time = time.time()

# Calculate and print the total time
print(f"Total running time: {end_time - start_time} seconds")

# ### Save Results

# In[8]:


# save final results
with open(os.path.join(args.output_dir, "coco_instances_results.json"), "w") as f:
    json.dump(results, f)

predictions = dict(
    [(k, {'image_id': -1, 'instances': []}) for k in range(len(scene_name_list))])
for idx in range(len(results)):
    id = results[idx]['image_id']
    predictions[scene_name_list.index('test_' + str(id).zfill(3))]['image_id'] = id

    predictions[scene_name_list.index('test_' + str(id).zfill(3))]['instances'].append(results[idx])

torch.save(predictions, os.path.join(args.output_dir, "instances_predictions.pth"))

print('Done!')


# ### Visualization

# In[9]:

# Random custom colors with a fixed random seed
random.seed(77)
thing_colors = []
for i in range(100):
    thing_colors.append(getColor())


# In[10]:


# Register Test Data for COCO evaluation
# test_path = "./test_data/test_4" # 1 for raw data, 2 for ratio=0.5, 4 for ratio=0.25, 8 for ratio=0.125  # test_4
# test_json = "./test_data/annotations/instances_test_4.json"  # instances_test_4
test_path = os.path.join(args.test_path, 'images')  # 1 for raw data, 2 for ratio=0.5, 4 for ratio=0.25, 8 for ratio=0.125
test_json = os.path.join(args.test_path, 'instances_test_4_'+scene_level+'.json')
register_coco_instances("coco_InsDet_test", {}, test_json, test_path)
MetadataCatalog.get("coco_InsDet_test").thing_colors = thing_colors

## evaluate the results using COCO API
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Load the ground truth COCO dataset
cocoGt = COCO(test_json)

# Load your detection results
cocoDt = cocoGt.loadRes(os.path.join(args.output_dir, "coco_instances_results.json"))

# Create a COCOeval object by initializing it with the ground truth and detection results
cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')

# Run the evaluation
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
