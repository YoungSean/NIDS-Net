from adapter import ModifiedClipAdapter, FeatureDataset, AdapterCNN, WeightAdapter
import torch
from torch.utils.data import DataLoader
import json
import os
from utils.instance_det_dataset import MaskedImageDataset
from torchvision import transforms


input_features = 1024
# feature_dataset = FeatureDataset(data_json='./obj_FFA/object_features_vitl14_reg.json', num_object=100)
# feature_dataset = FeatureDataset(data_json='./BOP_obj_feat/lmo_object_features.json', num_object=8)
# feature_dataset = FeatureDataset(data_json='./RoboTools_obj_feat/object_features.json', num_object=20)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = FeatureVectorModel(input_features, reduction=4, ratio=0.6).to(device) # Define your model
# adapter_args = 'mv24_demo5048_ratio_0.6_temp_0.05_epoch_40_lr_0.0001_bs_512_vec_reduction_4_L2e4_vitl_reg'
# model_path = 'adapter_weights/adapter2FC/'+adapter_args+'_weights.pth'
### bop challenge datasets
lmo_bop23_feature_dataset = FeatureDataset(data_json='bop23_obj_features/lmo/descriptors_cls_pbr.pth', num_object=8)
tless_bop23_feature_dataset = FeatureDataset(data_json='bop23_obj_features/tless/descriptors_cls_pbr.pth', num_object=30)
tudl_bop23_feature_dataset = FeatureDataset(data_json='bop23_obj_features/tudl/descriptors_cls_pbr.pth', num_object=3)
icbin_bop23_feature_dataset = FeatureDataset(data_json='bop23_obj_features/icbin/descriptors_cls_pbr.pth', num_object=2)
itodd_bop23_feature_dataset = FeatureDataset(data_json='bop23_obj_features/itodd/descriptors_cls_pbr.pth', num_object=28)
hb_bop23_feature_dataset = FeatureDataset(data_json='bop23_obj_features/hb/descriptors_cls_pbr.pth', num_object=33)
ycbv_bo23_feature_dataset = FeatureDataset(data_json='bop23_obj_features/ycbv/descriptors_cls_pbr.pth', num_object=21)

# adapter_args = 'bop_obj_shuffle_weight_0430_temp_0.05_epoch_500_lr_0.001_bs_32'
# adapter_args = "bop_obj_shuffle_0507_clip_temp_0.05_epoch_500_lr_0.0001_bs_32"
adapter_args = "bop_cls_obj_shuffle_0510_weight_temp_0.05_epoch_500_lr_0.001_bs_32"
# model_path = 'adapter_weights/adapter2FC/' + adapter_args + '_weights.pth'
model_path = f'adapter_weights/bop23/{adapter_args}_weights.pth'
# model_path = 'adapter_weights/adapter2FC/' + adapter_args + '.pt'
model = WeightAdapter(input_features, reduction=4).to('cuda')
# model = ModifiedClipAdapter(input_features).to('cuda')

model.load_state_dict(torch.load(model_path))

model.eval()  # Set the model to evaluation mode
# Assuming 'feature_dataset' is your Dataset object containing the feature vectors
# Assuming 'test_dataloader' is your DataLoader for the test dataset
batch_size = 16
dataset_folder = 'hb'
test_dataloader = DataLoader(hb_bop23_feature_dataset, batch_size=batch_size, shuffle=False)

adatped_features = []
for inputs, _ in test_dataloader:
    inputs = inputs.to(device)
    # labels = labels.to(device)
    with torch.no_grad():
        outputs = model(inputs)
        # Perform inference using the model
        # Your inference code here
        adatped_features.append(outputs)
adatped_features = torch.cat(adatped_features, dim=0)
feat_dict = dict()
feat_dict['features'] = adatped_features.detach().cpu().tolist()
# feat_dict['ffm_features'] = ffm_features.detach().cpu().tolist()
# output_dir = './adapted_obj_feats'
output_dir = f'./bop23_obj_features/{dataset_folder}'
adapter_type = 'weight'
json_filename = f'{adapter_type}_cls_obj_shuffle_0510_bs32_epoch_500_adapter_descriptors_pbr.json'
# json_filename = f'{adapter_args}.json'
with open(os.path.join(output_dir, json_filename), 'w') as f:
    json.dump(feat_dict, f)
print(f'Adapted features saved to {os.path.join(output_dir, json_filename)}')


# img_size = 448
# # Define transformations to be applied to the images
# transform = transforms.Compose([
#             transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ])
# masked_ins_det_dataset = MaskedImageDataset(data_dir='./database/Objects', json_path='./obj_FFA/object_features_vitl14_reg.json',
#                                             transform=transform, file_pattern='images/*g', mask_pattern='masks/*g')
# batch_size = 32
#
# save_features = True
# if save_features:
#     input_features = 1024
#     adapter_args = 'InsFew_ratio_0.6_temp_0.05_epoch_10_lr_0.0001_bs_512_L2e4_vitl_reg'
#     model_path = 'CNN_adapter_weights/' + adapter_args + '_weights.pth'
#     CNNadapter = AdapterCNN(input_features).to('cuda')
#     CNNadapter.load_state_dict(torch.load(model_path))
#     CNNadapter.eval()
#     batch_size = 32
#     test_dataloader = DataLoader(masked_ins_det_dataset, batch_size=batch_size, shuffle=False)
#
#     adatped_features = []
#     for masked_images, labels, features in test_dataloader:
#         masked_images = masked_images.to('cuda')
#         features = features.to('cuda')
#         with torch.no_grad():
#             outputs = CNNadapter(masked_images, features)
#             # Perform inference using the model
#             # Your inference code here
#             adatped_features.append(outputs)
#     adatped_features = torch.cat(adatped_features, dim=0)
#     feat_dict = dict()
#     feat_dict['features'] = adatped_features.detach().cpu().tolist()
#     # feat_dict['ffm_features'] = ffm_features.detach().cpu().tolist()
#     output_dir = './CNN_adapted_obj_feats'
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     json_filename = f'{adapter_args}.json'
#     with open(os.path.join(output_dir, json_filename), 'w') as f:
#         json.dump(feat_dict, f)
#     print(f"saving adapted features {json_filename}")