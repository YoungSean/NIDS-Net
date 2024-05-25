import numpy as np
import os
import json
import matplotlib.pyplot as plt
from matplotlib import colors
import torch
from utils.inference_utils import compute_similarity


data_json='./adapted_obj_feats/ins_0421_01_weight_temp_0.05_epoch_80_lr_0.002_bs_1024_vec_reduction_4_L2e4_vitl_reg.json'
with open(data_json, 'r') as f:
    feat_dict = json.load(f)


# with open(os.path.join('./obj_FFA', 'object_features.json'), 'r') as f:
#     feat_dict = json.load(f)

object_features = torch.Tensor(feat_dict['features']).cuda()

print(object_features.shape)

# for i in range(97, 98):
#     fig = plt.figure(figsize=(20, 8))  # Adjust the size as needed
#     for j in range(1, 11):
#         plt.subplot(2, 5, j)  # 2 rows, 5 columns, i-th plot
#         obj_id = 10 * i + j - 1
#         feat0 = object_features[24 * obj_id:24 * (obj_id + 1)]
#         # feat0 = object_features[10 * obj_id:10 * (obj_id + 1)]
#         sim_mat = compute_similarity(feat0, feat0)
#         # print(sim_mat)
#         # print(sim_mat.shape)
#         sim_array = sim_mat.detach().cpu().numpy()
#         # print("min: ", np.min(sim_array))
#         # visualize the similarity matrix
#         plt.title('Obj ID: ' + str(obj_id) + " min_cos: " + str(np.min(sim_array))[:5]+' mean_cos: '+str(np.mean(sim_array))[:5])
#         plt.imshow(sim_array)
#         plt.colorbar()
#
#     # Step 6: Adjust layout
#     plt.tight_layout()
#     # plt.savefig(os.path.join('precise_obj_feat_cos_sim','FFA_vitl','obj' + str(10 * i) + '_to_' + str(10 * i + 9) + '_CosSim.png'), dpi=300)
#     plt.show()




def plot_cos_sim_matrix(object_features, start_obj_id):
    Nr = 2
    Nc = 5

    fig, axs = plt.subplots(Nr, Nc, figsize=(20, 8))
    fig.suptitle('Cosine Similarity Matrix of Object Features_'+str(start_obj_id), fontsize=16)
    # fig.subplots_adjust(hspace=0.5, wspace=0.5)
    images = []
    for i in range(Nr):
        for j in range(Nc):
            # Generate data with a range that varies from one plot to the next.
            obj_id = 5 * i + j + start_obj_id
            feat0 = object_features[24 * obj_id:24 * (obj_id + 1)]
            # feat0 = object_features[10 * obj_id:10 * (obj_id + 1)]
            sim_mat = compute_similarity(feat0, feat0)
            # print(sim_mat)
            # print(sim_mat.shape)
            sim_array = sim_mat.detach().cpu().numpy()

            images.append(axs[i, j].imshow(sim_array))
            axs[i, j].label_outer()
            axs[i, j].set_title('Obj ID: ' + str(obj_id) + " min_cos: " + str(np.min(sim_array))[:5]+' mean_cos: '+str(np.mean(sim_array))[:5])

    # Find the min and max of all colors for use in setting the color scale.
    vmin = min(image.get_array().min() for image in images)
    vmax = max(image.get_array().max() for image in images)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    for im in images:
        im.set_norm(norm)

    # fig.colorbar(images[0], ax=axs, orientation='horizontal', fraction=.1)
    # plt.savefig(os.path.join('obj_feat_cos_sim_mat','FFA_vitl','obj' + str(start_obj_id) + '_to_' + str(start_obj_id + 9) + '_CosSim.png'), dpi=300)
    plt.tight_layout()
    plt.show()

for start in range(40, 50, 10):
    plot_cos_sim_matrix(object_features, start)
    print("Done for ", start)