'''
Convert original BOP challenge json file to instance json file for all scenes
for instance json file, every annotation (on specific instance) is an individual image, the image ids is the same as anno ids

reference: https://github.com/Jaraxxus-Me/VoxDet/blob/master/tools/aggregate_json.py
'''

import os
from re import L
from threading import local
from tqdm import tqdm, trange
import json
import shutil
from pycocotools.coco import COCO
import numpy as np
from glob import glob

dataset = 'RoboTools' #'RoboTools'# 'ycbv'
data_path = '../datasets'
if dataset == 'RoboTools':
    base_anno = os.path.join(data_path, dataset, 'test/000001/scene_gt_coco_ins.json')  # "datasets/ycbv/test/000048"
elif dataset == 'ycbv':
    base_anno = os.path.join(data_path, dataset, 'test/000048/scene_gt_coco.json')
filenames = sorted(os.listdir(os.path.join(data_path, dataset, 'test')))
scenes = []
for filename in filenames:
    if filename.startswith("00"):
        scenes.append(filename)
# scenes = glob(os.path.join(data_path, dataset, 'test', '000*'))
with open(base_anno, 'r') as f:
    anno_dict = json.load(f)

new_anno = anno_dict.copy()
new_anno["images"] = []
new_anno["annotations"] = []
new_im_id = 0
new_ann_id = 0
img_id_offset = 0 # maximum image id is around 2200 in ycbv dataset

generate_GT_json = False
if generate_GT_json:
    for scene in scenes:
        original_p2_anno = os.path.join(data_path, dataset, 'test/{:s}/current_scene_gt_coco.json'.format(scene))  # scene_gt_coco.json
        coco = COCO(original_p2_anno)

        for im in tqdm(coco.imgs.keys()):
            loriginal_image_info = coco.imgs[im]
            anns = coco.loadAnns(ids=coco.getAnnIds(im))
            local_img_info = loriginal_image_info.copy()
            ori_file = local_img_info['file_name']
            local_img_info['file_name'] = os.path.join(scene, ori_file)
            new_im_id = im + img_id_offset
            local_img_info['id'] = new_im_id
            for ann in anns:
                new_ann_id +=1
                ann['id'] = new_ann_id
                ann['image_id'] = new_im_id
                new_anno["annotations"].append(ann)
            new_anno["images"].append(local_img_info)
        img_id_offset += 3000 # maximum image id is around 2200 in ycbv dataset

    with open(os.path.join(data_path, dataset, 'test/scene_gt_coco_all_v2.json'), 'w') as f:
        json.dump(new_anno, f)

generate_pred_json = True
if generate_pred_json:
    modified_pred_annos = []
    img_id_offset_pred = 0
    for scene in scenes:
        if dataset == 'RoboTools':
            original_p2_anno = os.path.join(data_path, dataset, 'test/{:s}/current_scene_gt_coco.json'.format(scene))  # for RoboTools
        elif dataset == 'ycbv':
            original_p2_anno = os.path.join(data_path, dataset, 'test/{:s}/scene_gt_coco.json'.format(scene)) # for ycbv
        pred_annos = os.path.join(data_path, dataset, 'test/{:s}/weight_adapter_mv10k20epoch_samH_coco_instances_results_prediction.json'.format(scene)) # weight_adapter_
        coco = COCO(original_p2_anno)
        pred_annos = json.load(open(pred_annos))

        for i in trange(len(pred_annos)):
            img_id = pred_annos[i]['image_id']
            ann = pred_annos[i]
            new_ann_id +=1
            new_im_id = img_id + img_id_offset_pred
            ann['id'] = new_ann_id
            ann['image_id'] = new_im_id
            modified_pred_annos.append(ann)
        img_id_offset_pred += 3000 # maximum image id is around 2200 in ycbv dataset
    #
    with open(os.path.join(data_path, dataset, 'test/weight_adapter_mv10k20epoch_samH_coco_instances_results_prediction_all.json'), 'w') as f: # weight_adapter_
        json.dump(modified_pred_annos, f)

print('Total test images (split): {}'.format(new_im_id))
print('Total test instances (split): {}'.format(new_ann_id))