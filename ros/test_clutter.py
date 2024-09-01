import os
import json
import cv2
from nids_net import NIDS
import torch
from PIL import Image
import glob

#CLASS_NAMES = ["Background", "000_mustard", "001_spatula", "002_coconut_milk", "003_tazo_tea", "004_blue_sponge", "005_cards", "006_banana", "007_aluminum_foil", "008_red_car", "009_curry"]

# CLASS_NAMES = [str(i) for i in range(23)]
CLASS_NAMES = ["Background","001_original_coke", "002_sprite", "003_boston_lager", "004_diet_coke", "005_AW_root_beer", "006_zero_coke",
               "007_cranberry", "008_purple_power", "009_blue_power", "010_dr_pepper", "011_water", "012_fanta"] + [str(i) for i in range(23)]

# Model weights saved to adapter_weights/adapter2FC/ros_weight_0810_drink_snack_temp_0.05_epoch_80_lr_0.001_bs_1024_vec_reduction_4_weights.pth
# torch.Size([44, 1024])
# saving adapted features ./adapted_obj_feats/ros_weight_0810_drink_snack_temp_0.05_epoch_80_lr_0.001_bs_1024_vec_reduction_4.json
def test_ycb_sample():
    adapter_descriptors_path = "/home/yangxiao/Documents/NIDS-Net/adapted_obj_feats/ros_weight_0810_drink_snack_temp_0.05_epoch_80_lr_0.001_bs_1024_vec_reduction_4.json"
    with open(os.path.join(adapter_descriptors_path), 'r') as f:
        feat_dict = json.load(f)

    object_features = torch.Tensor(feat_dict['features']).cuda()
    object_features = object_features.view(-1, 2, 1024)
    weight_adapter_path = "adapter_weights/adapter2FC/ros_weight_0810_drink_snack_temp_0.05_epoch_80_lr_0.001_bs_1024_vec_reduction_4_weights.pth"
    model = NIDS(object_features, use_adapter=True, adapter_path=weight_adapter_path, gdino_threshold=0.4, class_labels=CLASS_NAMES)
    query_img_path = "/home/yangxiao/Documents/NIDS-Net/datasets/drinks_and_snacks/drink_clutter/color-000001.jpg"
    img = cv2.imread(query_img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    model.step(img, visualize=True)


    # list the image paths
    # folder_path = "/home/yangxiao/Documents/NIDS-Net/datasets/data_fetch_0725/data_fetch/scenes/0725T162628"
    # imgs = sorted(glob.glob(os.path.join(folder_path, "color-*.jpg")))
    # for query_img_path in imgs:
    #     #img_pil = Image.open(query_img_path)
    #     #img_pil.show()
    #     img = cv2.imread(query_img_path)
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #
    #     output_dir = "./ros/scene_results"
    #     if not os.path.exists(output_dir):
    #         os.makedirs(output_dir)
    #
    #     model.step(img, visualize=True, save_path=os.path.join(output_dir, os.path.basename(query_img_path)))
        # model.step(img, visualize=True)
        # break


if __name__ == "__main__":
    test_ycb_sample()