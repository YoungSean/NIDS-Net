from adapter import ModifiedClipAdapter, FeatureDataset, WeightAdapter
import torch
from torch.utils.data import DataLoader
import json
import os

input_features = 1024

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### bop challenge datasets
lmo_bop23_feature_dataset = FeatureDataset(data_json='datasets/bop23_challenge/datasets/templates_pyrender/lmo/descriptors_pbr.pth', num_object=8)
tless_bop23_feature_dataset = FeatureDataset(data_json='datasets/bop23_challenge/datasets/templates_pyrender/tless/descriptors_pbr.pth', num_object=30)
tudl_bop23_feature_dataset = FeatureDataset(data_json='datasets/bop23_challenge/datasets/templates_pyrender/tudl/descriptors_pbr.pth', num_object=3)
icbin_bop23_feature_dataset = FeatureDataset(data_json='datasets/bop23_challenge/datasets/templates_pyrender/icbin/descriptors_pbr.pth', num_object=2)
itodd_bop23_feature_dataset = FeatureDataset(data_json='datasets/bop23_challenge/datasets/templates_pyrender/itodd/descriptors_pbr.pth', num_object=28)
hb_bop23_feature_dataset = FeatureDataset(data_json='datasets/bop23_challenge/datasets/templates_pyrender/hb/descriptors_pbr.pth', num_object=33)
ycbv_bo23_feature_dataset = FeatureDataset(data_json='datasets/bop23_challenge/datasets/templates_pyrender/ycbv/descriptors_pbr.pth', num_object=21)


adapter_args = "bop_obj_shuffle_0529_weight_temp_0.05_epoch_500_lr_0.001_bs_32"
model_path = f'adapter_weights/bop23/{adapter_args}_weights.pth'
model = WeightAdapter(input_features, reduction=4).to('cuda')
# model = ModifiedClipAdapter(input_features).to('cuda')

model.load_state_dict(torch.load(model_path))

model.eval()  # Set the model to evaluation mode

batch_size = 16
dataset_folder = 'lmo'
test_dataloader = DataLoader(lmo_bop23_feature_dataset, batch_size=batch_size, shuffle=False)

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
output_dir = f'./bop23_obj_features/{dataset_folder}'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
adapter_type = 'weight'
json_filename = f'{adapter_type}_obj_shuffle_0529_bs32_epoch_500_adapter_descriptors_pbr.json'
# json_filename = f'{adapter_args}.json'
with open(os.path.join(output_dir, json_filename), 'w') as f:
    json.dump(feat_dict, f)
print(f'Adapted features saved to {os.path.join(output_dir, json_filename)}')

