import numpy as np
import os
import json
import matplotlib.pyplot as plt
from matplotlib import colors
import torch
from torch import nn
from utils.inference_utils import compute_similarity

# './obj_FFA', 'object_features.json'

# with open(os.path.join('./obj_FFA', 'FeatUp_cropped_object_features_vits_img224.json'), 'r') as f:
#     feat_dict = json.load(f)

output_dir = './adapted_obj_feats'
json_filename = 'adapted_obj_features_ratio_0.6_temp_0.1_epoch_100_lr_0.001_bs_512_vitl_reg.json'
# os.path.join('./obj_FFA', 'object_features.json')
with open(os.path.join(output_dir, json_filename), 'r') as f:
    feat_dict = json.load(f)

object_features = torch.Tensor(feat_dict['features']).cuda()
object_features = nn.functional.normalize(object_features, dim=1, p=2)

same_object_similarity = False
if same_object_similarity:
    # plt.subplot(2, 5, j)  # 2 rows, 5 columns, i-th plot
    avg_sim_array = np.zeros(100)  # 100 objects
    robust_obj_features = []
    topk = 13
    num_obj = 24
    for i in range(0, 100):
        obj_id = i
        feat0 = object_features[num_obj * obj_id:num_obj * (obj_id + 1)]
        sim_mat = compute_similarity(feat0, feat0)
        avg_sim = torch.mean(sim_mat, dim=-1)
        obj_avg_sim = torch.mean(avg_sim)
        obj_avg_sim = obj_avg_sim.detach().cpu().numpy()
        avg_sim_array[obj_id] = obj_avg_sim
        sim_array = sim_mat.detach().cpu().numpy()

else:
    # plt.subplot(2, 5, j)  # 2 rows, 5 columns, i-th plot
    avg_sim_array = np.zeros(99)  # 100 objects
    robust_obj_features = []
    topk = 13
    num_obj = 24
    for i in range(0, 99):
        obj_id = i
        feat0 = object_features[num_obj * obj_id:num_obj * (obj_id + 1)]
        feat1 = object_features[num_obj * (obj_id+1):num_obj * (obj_id + 2)]
        sim_mat = compute_similarity(feat0, feat1)
        avg_sim = torch.mean(sim_mat, dim=-1)
        obj_avg_sim = torch.mean(avg_sim)
        obj_avg_sim = obj_avg_sim.detach().cpu().numpy()
        avg_sim_array[obj_id] = obj_avg_sim
        vals, idxs = torch.topk(avg_sim, k=topk) # top 5 robust object features
        # get new object features from the topk similar object features
        # print(idxs[:-1])
        # print(idxs[1:])
        # print(feat0[idxs[:-1], :][:3,:5])
        # print(feat0[idxs[1:], :][:3,:5])
        # generate_obj_features = (feat0[idxs[:-1], :] + feat0[idxs[1:], :]) / 2.0 # generate k-1 robust object features
        # # get new features from object features
        # # print(generate_obj_features[0,:2])
        # # generate_obj_features = (feat0[:-1, :] + feat0[1:, :]) / 2.0
        # robust_obj_features.append(feat0)
        # robust_obj_features.append(generate_obj_features)

        sim_array = sim_mat.detach().cpu().numpy()
        # print("min: ", np.min(sim_array))
        # print("avg_sim: ", avg_sim_array)
        # # print("obj_avg_sim: ", obj_avg_sim.detach().cpu().numpy())
        # print("top5: ", idxs)
        # print(feat0[idxs, :].shape)
print("avg_sim_array: ", avg_sim_array)
mean_avg_sim = np.mean(avg_sim_array)
print("mean_avg_sim: ", mean_avg_sim)
#
# robust_obj_features = torch.cat(robust_obj_features, dim=0)
# print("robust_obj_features: ", robust_obj_features.shape)
# feat_dict = dict()
# feat_dict['features'] = robust_obj_features.detach().cpu().tolist()
# with open(os.path.join('./obj_FFA', 'object_features_add_plain'+str(23)+'.json'), 'w') as f:
#     json.dump(feat_dict, f)

##### Plot the histogram, density plot, and boxplot

import matplotlib.pyplot as plt
import seaborn as sns

visualize = True
if visualize:
    data = avg_sim_array

    # Set up the matplotlib figure
    plt.figure(figsize=(15, 5))

    # Plot a histogram
    plt.subplot(1, 3, 1)
    plt.hist(data, bins=10, alpha=0.7, color='blue')
    plt.title('Histogram')

    # Plot a density plot
    plt.subplot(1, 3, 2)
    sns.kdeplot(data, shade=True, color='green')
    plt.title('Density Plot')

    # Plot a boxplot
    plt.subplot(1, 3, 3)
    sns.boxplot(data, orient='v', width=0.3)
    plt.title('Boxplot')

    # Show the plots
    plt.tight_layout()
    plt.show()


# vals, idxs = torch.topk(avg_sim, k=5)
# sim_array = sim_mat.detach().cpu().numpy()
# avg_sim_array = avg_sim.detach().cpu().numpy()
# print("min: ", np.min(sim_array))
# print("avg_sim: ", avg_sim_array)
# print("obj_avg_sim: ", obj_avg_sim.detach().cpu().numpy())
# print("top5: ", idxs)
# print(feat0[idxs,:].shape)
#
# print("min: ", np.min(sim_array))
# # visualize the similarity matrix
# plt.title('Obj ID: ' + str(obj_id) + " min_cos: " + str(np.min(sim_array))[:5])
# plt.imshow(sim_array)
# plt.colorbar()