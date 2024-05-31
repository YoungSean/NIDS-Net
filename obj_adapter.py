from adapter_dataset import ObjectFeatureDataset
import torch

from torch.utils.data import DataLoader, ConcatDataset
import os
import json
import hydra
from omegaconf import DictConfig, OmegaConf
from adapter import ModifiedClipAdapter, WeightAdapter, InfoNCELoss
from hydra.core.hydra_config import HydraConfig

@hydra.main(config_path="configs", config_name="config")
def main(cfg : DictConfig):
    original_cwd = HydraConfig.get().runtime.cwd
    print(original_cwd)
    combine_dataset = True
    adapter_type = cfg.params.adapter_type
    dataset_name = f'bop_obj_shuffle_0529_{adapter_type}'
    # dataset_name = f'lmo_{adapter_type}'
    temperature = cfg.params.temperature
    batch_size = cfg.params.batch_size
    shuffle = cfg.params.shuffle

    ### bop challenge datasets
    print(os.getcwd())
    lmo_bop23_feature_dataset = ObjectFeatureDataset(data_json=f'{original_cwd}/datasets/bop23_challenge/datasets/templates_pyrender/lmo/descriptors_pbr.pth',
                                                     num_object=8)
    tless_bop23_feature_dataset = ObjectFeatureDataset(data_json=f'{original_cwd}/datasets/bop23_challenge/datasets/templates_pyrender/tless/descriptors_pbr.pth',
                                                       num_object=30, label_offset=8)
    tudl_bop23_feature_dataset = ObjectFeatureDataset(data_json=f'{original_cwd}/datasets/bop23_challenge/datasets/templates_pyrender/tudl/descriptors_pbr.pth',
                                                      num_object=3, label_offset=38)
    icbin_bop23_feature_dataset = ObjectFeatureDataset(data_json=f'{original_cwd}/datasets/bop23_challenge/datasets/templates_pyrender/icbin/descriptors_pbr.pth',
                                                       num_object=2, label_offset=41)
    itodd_bop23_feature_dataset = ObjectFeatureDataset(data_json=f'{original_cwd}/datasets/bop23_challenge/datasets/templates_pyrender/itodd/descriptors_pbr.pth',
                                                       num_object=28, label_offset=43)
    hb_bop23_feature_dataset = ObjectFeatureDataset(data_json=f'{original_cwd}/datasets/bop23_challenge/datasets/templates_pyrender/hb/descriptors_pbr.pth',
                                                    num_object=33, label_offset=71)
    ycbv_bo23_feature_dataset = ObjectFeatureDataset(data_json=f'{original_cwd}/datasets/bop23_challenge/datasets/templates_pyrender/ycbv/descriptors_pbr.pth',
                                                     num_object=21, label_offset=104)

    cur_feature_dataset = hb_bop23_feature_dataset

    # Example training loop
    input_features = cfg.params.input_features  # Size of the input feature vector
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if adapter_type == 'clip':
        learning_rate = 1e-4
        model = ModifiedClipAdapter(input_features).to(device)
    else:
        learning_rate = 1e-3
        model = WeightAdapter(input_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)  #
    criterion = InfoNCELoss(temperature=temperature).to(device)
    epochs = cfg.params.epochs
    if combine_dataset:
        combined_dataset = ConcatDataset(
            [lmo_bop23_feature_dataset, tless_bop23_feature_dataset, tudl_bop23_feature_dataset,
             icbin_bop23_feature_dataset, itodd_bop23_feature_dataset, hb_bop23_feature_dataset,
             ycbv_bo23_feature_dataset])
        dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        dataloader = DataLoader(cur_feature_dataset, batch_size=batch_size, shuffle=shuffle)

    for epoch in range(epochs):
        for inputs, labels in dataloader:  # in dataloader: tqdm(dataloader)
            inputs = inputs.view(-1, input_features).to(device)
            labels = labels.view(-1).to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

    save_model = True
    adapter_args = f'{dataset_name}_temp_{temperature}_epoch_{epochs}_lr_{learning_rate}_bs_{batch_size}'
    if save_model:
        # Assuming your model is named 'model'
        model_output_dir = f'{original_cwd}/adapter_weights/bop23'
        os.makedirs(model_output_dir, exist_ok=True)
        model_path = f'{model_output_dir}/{adapter_args}_weights.pth'  # Define the path where you want to save the model
        torch.save(model.state_dict(), model_path)

        print(f'Model weights saved to {model_path}')

    save_features = False
    if save_features:
        # Assuming model is already defined and loaded with trained weights
        model.eval()  # Set the model to evaluation mode
        batch_size = 64
        # Assuming 'feature_dataset' is your Dataset object containing the feature vectors
        # Assuming 'test_dataloader' is your DataLoader for the test dataset
        test_dataloader = DataLoader(cur_feature_dataset, batch_size=batch_size, shuffle=False)

        adatped_features = []
        for inputs, labels in test_dataloader:
            inputs = inputs.view(-1, input_features).to(device)
            # labels = labels.to(device)
            with torch.no_grad():
                outputs = model(inputs)
                # Perform inference using the model
                # Your inference code here
                adatped_features.append(outputs)
        adatped_features = torch.cat(adatped_features, dim=0)
        print(adatped_features.size())
        feat_dict = dict()
        feat_dict['features'] = adatped_features.detach().cpu().tolist()
        # feat_dict['ffm_features'] = ffm_features.detach().cpu().tolist()
        # output_dir = f'./bop23_obj_features/{dataset_folder}'
        output_dir = f'{original_cwd}/adapted_obj_feats'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        json_filename = f'{adapter_args}.json'
        # json_filename = f'{adapter_type}_bs1024_epoch_{epochs}_adapter_descriptors_pbr.json'
        with open(os.path.join(output_dir, json_filename), 'w') as f:
            json.dump(feat_dict, f)
        print(f"saving adapted features {os.path.join(output_dir, json_filename)}")

if __name__ == '__main__':
    main()


