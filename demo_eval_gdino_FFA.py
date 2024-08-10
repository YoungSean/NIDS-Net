#!/usr/bin/env python
# coding: utf-8


# Testing:
# 1. Object Proposal Generation by Grounded SAM
# 2. Feature Extraction for proposals by DINOv2
# 3. Proposal/Instance Matching -- Stable Matching

# In[6]:

import time
import argparse
import cv2
import glob
import json
import logging
import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torchvision import transforms as pth_transforms
import torchvision.transforms as T

sys.path.append("../detectron2")
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances


# sys.path.append("../dinov2-main")
# from dinov2.eval.setup import get_args_parser as get_setup_args_parser


sys.path.append(".")
from utils.visualizer import ColorMode, Visualizer
from utils.instance_det_dataset import RealWorldDataset
from utils.inference_utils import compute_similarity, stableMatching, get_bbox_masks_from_gdino_sam, \
    get_object_proposal, getColor, create_instances, nms, apply_nms, get_features, get_cls_token
from tqdm import trange
from adapter import ModifiedClipAdapter, WeightAdapter
from utils.img_utils import get_masked_image

# logger = logging.getLogger("dinov2")


def get_args_parser(
        description: Optional[str] = None,
        parents: Optional[List[argparse.ArgumentParser]] = [],
        add_help: bool = True,
):
    #setup_args_parser = get_setup_args_parser(parents=parents, add_help=False)
    parents = []

    parser = argparse.ArgumentParser(
        description=description,
        parents=parents,
        add_help=add_help,
    )
    parser.add_argument(
        "--train_path",
        default="../database/train",
        type=str,
        help="Path to train dataset.",
    )
    parser.add_argument(
        "--test_path",
        default="../database/test",
        type=str,
        help="Path to test dataset.",
    )
    parser.add_argument(
        "--imsize",
        default=448,
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


# Default args and initialize model
args_parser = get_args_parser(description="Grounded SAM-DINOv2 Instance Detection")
imsize = 448
tag = "mask"  # bbox
img_id = 39
args = args_parser.parse_args(args=["--train_path", "database/Objects",
                                    "--test_path", "test_data/test_1/test_"+str(img_id).zfill(3)+".jpg",  # test_002
                                    "--output_dir", "exps/demo0501_" + str(imsize) + "_" + tag,
                                    ])
os.makedirs(args.output_dir, exist_ok=True)


# use dino v2 to extract features
encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder.to(device)
encoder.eval()

# Assuming the model's architecture is defined in 'FeatureVectorModel' class
use_adapter = False
adapter_type = "weight"
if use_adapter:
    input_features = 1024
    if adapter_type == "clip":
        adapter_args = 'Ins_ratio_0.6_temp_0.05_epoch_40_lr_0.001_bs_1024_vec_reduction_4_L2e4_vitl_reg'
        model_path = 'adapter_weights/adapter2FC/'+adapter_args+'_weights.pth'
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



# In[9]:

output_dir = './obj_FFA'
#json_filename = 'object_features_vitl14_reg.json'
json_filename = "object_features_l_reg_class.json"
if use_adapter:
    output_dir = './adapted_obj_feats'
    json_filename = adapter_args+'.json'


with open(os.path.join(output_dir, json_filename), 'r') as f:
    feat_dict = json.load(f)


object_features = torch.Tensor(feat_dict['features']).to(device)
object_features = nn.functional.normalize(object_features, dim=1, p=2)
print("object_features: ", object_features.shape)

# In[10]:
from absl import app, logging
from PIL import Image as PILImg


from robokit.ObjDetection import GroundingDINOObjectPredictor, SegmentAnythingPredictor

# Path to the input image
image_path = args.test_path


logging.info("Initialize object detectors")
gdino = GroundingDINOObjectPredictor(use_vitb=False, threshold=0.15)
SAM = SegmentAnythingPredictor(vit_model="vit_h")
# SAM = SegmentAnythingPredictor(vit_model="vit_t") # use mobile sam

logging.info("Open the image and convert to RGB format")
image_pil = PILImg.open(image_path).convert("RGB")

# #### 1. Get object proposals with Grounded-SAM
accurate_bboxs, masks = get_bbox_masks_from_gdino_sam(image_path, gdino, SAM, text_prompt='objects', visualize=False)


# #### 2. Crop proposals from the high-resolution version

# In[11]:

start_time = time.time()
scene_features_list = []
proposals_list = []
scene_name_list = []
mask = masks.cpu().numpy()
accurate_bboxs = accurate_bboxs.cpu().numpy()
scene_name = os.path.basename(image_path).split('.')[0]
scene_name_list.append(scene_name)
rois, sel_rois, cropped_imgs, cropped_masks = get_object_proposal(args.test_path, accurate_bboxs, masks, tag=tag, ratio=0.25, save_rois=False, output_dir=args.output_dir, save_proposal=False)
scene_features = []
for i in trange(len(cropped_imgs)):
    img = cropped_imgs[i]
    mask = cropped_masks[i]
    #ffa_feature= get_features([img], [mask], encoder,device=device, img_size=imsize)
    ffa_feature = get_cls_token([img], [mask], encoder, device=device, img_size=imsize)
    with torch.no_grad():
        if use_adapter:
            ffa_feature = adapter(ffa_feature)
    scene_features.append(ffa_feature)
scene_features = torch.cat(scene_features, dim=0)
scene_features = nn.functional.normalize(scene_features, dim=1, p=2)

scene_features_list.append(scene_features)
# total_proposals[scene_name] = sel_rois
proposals_list.append(sel_rois)
# rgb_normalize = T.Compose(
#             [
#                 T.ToTensor(),
#                 T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#             ]
#         )
# masks = masks.squeeze(1).to(torch.float32)
# rois, sel_rois, cropped_imgs, cropped_masks = get_object_proposal_tensor(args.test_path, accurate_bboxs, masks, img_size=448, rgb_normalize=rgb_normalize,tag=tag, ratio=0.25, save_rois=True, output_dir=args.output_dir)
# scene_features = []
#

# batch_size = 32  # Define the batch size
#
# dataset = TensorDataset(cropped_imgs, cropped_masks)
# # Create a DataLoader
# data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
#
# # Iterate over the DataLoader
# for image_batch, mask_batch in data_loader:
#     ffa_feature = get_features_via_batch_tensor(image_batch, mask_batch, encoder, img_size=imsize)
#     if use_adapter:
#         ffa_feature = adapter(ffa_feature)
#     # Extend the main feature list with the features from this batch
#     scene_features.append(ffa_feature)
# scene_features = torch.cat(scene_features, dim=0)
# scene_features = nn.functional.normalize(scene_features, dim=1, p=2)
# Capture the end time
end_time = time.time()

# Calculate and print the total time
print(f"Total running time: {end_time - start_time} seconds")

# image_height, image_width = raw_image.shape[:-1]
scene_name = os.path.basename(args.test_path).split('.')[0]



# In[13]:


# transform = pth_transforms.Compose([pth_transforms.ToTensor(),])
#
# # for demo
# scene_dataset = RealWorldDataset(args.output_dir, scene_name, data=rois, transform=transform, imsize=args.imsize)
# scene_features = get_scene_feature(args.output_dir, scene_name, scene_dataset, model, args.batch_size, args.num_workers)


# ####  3. Compute Cosine Similarity and Proposal/Instance Matching

# In[7]:

# reshape scene feature matrix
# scene_cnt = [0, *scene_dataset.cfg['length']]
# scene_idx = [sum(scene_cnt[:i + 1]) for i in range(len(scene_cnt))]
# scene_features_list = [scene_features[scene_idx[i]:scene_idx[i + 1]] for i in
#                         range(len(scene_dataset.cfg['length']))]
#
# proposals = scene_dataset.cfg['proposals']
# proposals_list = [proposals[scene_idx[i]:scene_idx[i + 1]] for i in range(len(scene_dataset.cfg['length']))]

num_object = 100 # len(object_dataset.cfg['obj_name'])
num_example = len(object_features) // num_object

results = []

for idx, scene_feature in enumerate(scene_features_list):
    sim_mat = compute_similarity(object_features, scene_feature)
    sim_mat = sim_mat.view(len(scene_feature), num_object, num_example)
    sims, _ = torch.max(sim_mat, dim=2)  # choose max score over profile examples of each object instance
    max_ins_sim, initial_result = torch.max(sims, dim=1)

    proposals = proposals_list[idx]
    num_proposals = len(proposals)

    ########################################## Stable Matching Strategy ##########################################
    do_matching = True
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
        print("Done!")

        # ------------ thresholding ------------
        preds = Matches.copy()
        # for key, value in Matches.items():
        #     if sel_sims_symmetric[int(sel_roi_ids.index(key)), int(sel_obj_ids.index(value))] <= score_thresh_predefined:
        #         del preds[key]
        #         continue
        
        # ------------ save per scene results ------------

        for k, v in preds.items():
            if int(k) >= num_proposals:  # since the number of proposals is less than the number of object features
                break

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
        THRESHOLD_OBJECT_SCORE = 0.6
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

        
    # print("Done!")


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
test_path = "./test_data/test_4" # 1 for raw data, 2 for ratio=0.5, 4 for ratio=0.25, 8 for ratio=0.125  # test_4
# test_json = "./test_data/annotations/instances_test_4.json"  # instances_test_4
test_json = "./test_data/annotations/single_image_ground_truth_"+str(img_id)+".json"  # instances_test_4
register_coco_instances("coco_InsDet_test", {}, test_json, test_path)
MetadataCatalog.get("coco_InsDet_test").thing_colors = thing_colors


# In[11]:


pred_dir = os.path.join(args.output_dir,"predictions")
os.makedirs(pred_dir, exist_ok=True)
test_metadata = MetadataCatalog.get("coco_InsDet_test")
test_dataset_dicts = DatasetCatalog.get("coco_InsDet_test")

idx = scene_name_list.index(scene_name)
# d = test_dataset_dicts[int(scene_name.split('_')[-1])]
d = test_dataset_dicts[0] # idx
img = cv2.imread(d["file_name"])
base_name = os.path.basename(d["file_name"]).split(".")[0]

# visualize GT
visGT = Visualizer(img[:, :, ::-1],
                   metadata=test_metadata,
                   scale=1.0,
                   instance_mode=ColorMode.SEGMENTATION)
vis_gt = visGT.draw_dataset_dict(d)
cv2.imwrite(os.path.join(pred_dir, base_name + "_gt.jpg"),
            vis_gt.get_image()[:, :, ::-1])

# visualize pred
# predictions = torch.load(os.path.join(args.output_dir, "instances_predictions.pth"))
pred = create_instances(predictions[idx]['instances'], img.shape[:2], test_metadata)
keep_ids = apply_nms(pred)  # apply NMS

visPred = Visualizer(img[:, :, ::-1],
                     metadata=test_metadata,
                     scale=1.0,
                     instance_mode=ColorMode.SEGMENTATION)
vis_pred = visPred.draw_instance_predictions(pred.to("cpu"), keep_ids)
cv2.imwrite(os.path.join(pred_dir, base_name + "_pred_SAM+DINOv2.jpg"),
            vis_pred.get_image()[:, :, ::-1])

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Load the ground truth COCO dataset
cocoGt = COCO('test_data/annotations/single_image_ground_truth_'+str(img_id)+'.json')

# Load your detection results
cocoDt = cocoGt.loadRes(os.path.join(args.output_dir, "coco_instances_results.json"))

# Create a COCOeval object by initializing it with the ground truth and detection results
cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')

# Run the evaluation
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()

# 'bbox' specifies that we are doing evaluation for bounding box detection.
# You can change it to 'segm' for segmentation tasks.


# In[12]:


fig = plt.figure(figsize=(30,60), dpi=64, facecolor='w', edgecolor='k')
ax = plt.subplot(211)
ax.set_title("GroundTruth")
plt.imshow(vis_gt.get_image()[:, :, :])
ax = plt.subplot(212)
ax.set_title("Predictions on SAM-DINOv2")
plt.imshow(vis_pred.get_image()[:, :, :])
plt.tight_layout()
# plt.savefig('./result_images/InsDet/'+base_name+'_SAM_DINOv2.png', dpi=300)
plt.show()

