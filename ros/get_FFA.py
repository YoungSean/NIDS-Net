import os
import json
import cv2
from nids_net import NIDS
import torch
from PIL import Image
import glob

data_dir = "/home/yangxiao/Documents/NIDS-Net/datasets/data_fetch_0725/data_fetch/objects"
obj_folder_pattern = "00*"
source_list = sorted(glob.glob(os.path.join(data_dir, obj_folder_pattern)))
print(source_list)

model = NIDS(None, use_adapter=False, gdino_threshold=0.4)
object_features = []
for folder in source_list:
    imgs = sorted(glob.glob(os.path.join(folder, "color-*.jpg")))
    template_embeddings = []
    for img_path in imgs:
        img_pil = Image.open(img_path)
        ffa_features = model.get_template_feature_per_image(img_pil) # ffa_features: torch.Size([1, 1, 1024])
        ffa_features = ffa_features.squeeze(0) # 1, 1, 1024 -> 1, 1024
        template_embeddings.append(ffa_features)
    template_embeddings = torch.cat(template_embeddings, dim=0)
    object_features.append(template_embeddings)
object_features = torch.cat(object_features, dim=0)

# Save the object features
feat_dict = dict()
feat_dict['features'] = object_features.detach().cpu().tolist()
output_dir = "./ros/object_features"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

json_filename = 'object_features_10.json'
with open(os.path.join(output_dir, json_filename), 'w') as f:
    json.dump(feat_dict, f)