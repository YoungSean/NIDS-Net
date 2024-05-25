import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
import json
from torch import nn
import os

output_dir = './obj_FFA'
json_filename = 'fewsol_object_features_vitl14_reg.json' #'object_features.json'
# output_dir = './adapted_obj_feats'
# json_filename = 'adapted_obj_features_ratio_0.6_temp_0.1_epoch_100_lr_0.001_bs_512_vitl_reg.json'
# os.path.join('./obj_FFA', 'object_features.json')
#
with open(os.path.join(output_dir, json_filename), 'r') as f:
    feat_dict = json.load(f)

object_features = torch.Tensor(feat_dict['features']).cuda()
object_features = nn.functional.normalize(object_features, dim=1, p=2)
# Assuming 'feature_vectors' is your PyTorch tensor of shape (2400, 1024)
feature_vectors_np = object_features.detach().cpu().numpy()  # Convert to NumPy array

# Apply t-SNE to reduce to 2 dimensions for visualization
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(feature_vectors_np)

# Generate labels for each point based on its class (24 feature vectors per class)
labels = np.repeat(np.arange(feature_vectors_np.shape[0] // 24), 24)

# Plot the t-SNE results with matplotlib
plt.figure(figsize=(20, 25))
for class_id in np.unique(labels):
    indices = np.where(labels == class_id)
    plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], label=f'Class {class_id}', alpha=0.5)
plt.legend()
plt.title('t-SNE visualization of feature vectors')
plt.xlabel('t-SNE axis 1')
plt.ylabel('t-SNE axis 2')
plt.show()
