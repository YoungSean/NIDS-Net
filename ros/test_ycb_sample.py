import os
import json
import cv2
from nids_net import NIDS
import torch
from PIL import Image

def test_ycb_sample():
    # use ros/000047.png as the sample image to test the model
    img_path = "ros/000047.png"
    query_img_path = "ros/000047.png"
    #img_path = "/home/yangxiao/Documents/datasets/some_objects/data_fetch/object1/color-000000.jpg"
    #query_img_path = "/home/yangxiao/Documents/datasets/some_objects/data_fetch/object1/color-000004.jpg"
    img_pil = Image.open(img_path)
    #img_pil.show()
    img = cv2.imread(query_img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    adapter_descriptors_path = "ros/weight_obj_shuffle2_0501_bs32_epoch_500_adapter_descriptors_pbr.json"
    with open(os.path.join(adapter_descriptors_path), 'r') as f:
        feat_dict = json.load(f)

    object_features = torch.Tensor(feat_dict['features']).cuda()
    object_features = object_features.view(-1, 42, 1024)
    weight_adapter_path = "ros/bop_obj_shuffle_weight_0430_temp_0.05_epoch_500_lr_0.001_bs_32_weights.pth"
    model = NIDS(object_features, use_adapter=True, adapter_path=weight_adapter_path, gdino_threshold=0.2)
    #model.get_template_feature_per_image(img_pil)
    model.step(img, visualize=True)


if __name__ == "__main__":
    test_ycb_sample()