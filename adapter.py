
import torch.nn.functional as F

from utils.adapter_dataset import FeatureDataset
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import json
from tqdm import trange, tqdm



# reference: https://github.com/gaopengcuhk/CLIP-Adapter/blob/main/clip_adapter.py#L55C1-L67C17
class ClipAdapter(nn.Module):
    """
    Original clip adapter.
    """
    def __init__(self, c_in, reduction=4):
        super(ClipAdapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x

class ModifiedClipAdapter(nn.Module):
    """
    Modified version of the CLIP adapter for better performance.
    Add Dropout layer.
    """
    def __init__(self, c_in, reduction=4, ratio=0.6):
        super(ModifiedClipAdapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )
        self.ratio = ratio

    def forward(self, inputs):
        inputs = F.normalize(inputs, dim=-1, p=2)
        x = self.fc(inputs)
        x = self.ratio * x + (1 - self.ratio) * inputs
        return x

class WeightAdapter(nn.Module):
    """
    Predict weights for each feature vector.
    """
    def __init__(self, c_in, reduction=4, scalar=10.0):
        """

        @param c_in: The channel size of the input feature vector
        @param reduction: the reduction factor for the hidden layer
        @param scalar: A scalar to scale the input feature vector
        """
        super(WeightAdapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )
        # self.ratio = ratio
        self.scalar = scalar

    def forward(self, inputs):
        # inputs = F.normalize(inputs, dim=-1, p=2)
        inputs = self.scalar * inputs
        x = self.fc(inputs)
        x = x.sigmoid()
        x = x * inputs

        return x


# modified from SimCLR loss: https://github.com/sthalles/SimCLR/blob/master/simclr.py#L26
class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.5, eps=1e-8):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.eps = eps  # Small constant to avoid division by zero or log(0)

    def forward(self, features, labels):
        # original_labels = labels
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)
        labels = labels.contiguous().view(-1, 1)
        mask_positive = torch.eq(labels, labels.T).float()
        mask_negative = 1 - mask_positive
        mask_positive.fill_diagonal_(0)

        exp_logits = torch.exp(similarity_matrix / self.temperature)
        sum_negatives = torch.sum(exp_logits * mask_negative, dim=1, keepdim=True) + self.eps # replace 1-mask_positive with mask_negative
        sum_positives = torch.sum(exp_logits * mask_positive, dim=1, keepdim=True)

        # Adding eps inside the log to avoid log(0)
        loss = -torch.log(sum_positives / (sum_positives + sum_negatives) + self.eps)
        loss = loss.mean()

        return loss


if __name__ == '__main__':
    adapter_type = 'weight'
    dataset_name = f'ros_{adapter_type}_0810_drink_snack'
    temperature = 0.05
    ratio = 0.6
    feature_dataset = FeatureDataset(data_json='./ros/object_features/drink_snack_features.json', num_object=22) # 100 objects in total
    # Assuming 'features' is your (N, 1024) tensor
    batch_size = 1024

    # robo_feature_dataset = FeatureDataset(data_json='./RoboTools_obj_feat/object_features.json', num_object=20) # 20 objects in total
    # ycbv_feature_dataset = FeatureDataset(data_json='./BOP_obj_feat/ycbv_object_features.json', num_object=21) # 21 objects in total
    # lmo_feature_dataset = FeatureDataset(data_json='./BOP_obj_feat/lmo_object_features.json', num_object=8)


    cur_feature_dataset = feature_dataset

    # Example training loop
    input_features = 1024  # Size of the input feature vector, 1024 for large, 768 for base, 384 for small
    reduction = 4 # Reduction factor for the hidden layer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if adapter_type == 'clip':
        learning_rate = 1e-4
        model = ModifiedClipAdapter(input_features, reduction=reduction, ratio=ratio).to(device)
    else:
        learning_rate = 1e-3
        model = WeightAdapter(input_features, reduction=reduction).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4) #
    criterion = InfoNCELoss(temperature=temperature).to(device)
    epochs = 80

    dataloader = DataLoader(cur_feature_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(epochs):

        for inputs, labels in dataloader: # in dataloader: tqdm(dataloader)
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')


    save_model = True
    adapter_args = f'{dataset_name}_temp_{temperature}_epoch_{epochs}_lr_{learning_rate}_bs_{batch_size}_vec_reduction_{reduction}'
    os.makedirs('adapter_weights/adapter2FC', exist_ok=True)
    if save_model:
        model_path = f'adapter_weights/adapter2FC/{adapter_args}_weights.pth'
        # Save the model state dictionary
        torch.save(model.state_dict(), model_path)

        print(f'Model weights saved to {model_path}')

    save_features = True
    if save_features:
        # Assuming model is already defined and loaded with trained weights
        model.eval()  # Set the model to evaluation mode
        batch_size = 64
        test_dataloader = DataLoader(cur_feature_dataset, batch_size=batch_size, shuffle=False)

        adatped_features = []
        for inputs, labels in test_dataloader:
            inputs = inputs.to(device)
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
        output_dir = './adapted_obj_feats'
        os.makedirs(output_dir, exist_ok=True)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        json_filename = f'{adapter_args}.json'
        with open(os.path.join(output_dir, json_filename), 'w') as f:
            json.dump(feat_dict, f)
        print(f"saving adapted features {os.path.join(output_dir, json_filename)}")


