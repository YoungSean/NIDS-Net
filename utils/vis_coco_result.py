from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import json
import cv2
import os

# Initialize COCO api for instance annotations (replace 'path/to/...' with actual file paths)
# for robotools
# folder_path = '../datasets/RoboTools/test/000014'
# coco_gt = COCO(os.path.join(folder_path, 'current_scene_gt_coco.json'))
# coco_dt = coco_gt.loadRes(os.path.join(folder_path,'weight_adapter_80epoch_samH_coco_instances_results_prediction.json'))

# for YCBV
# folder_path = '../datasets/ycbv/test/000058'
# coco_gt = COCO(os.path.join(folder_path, 'scene_gt_coco.json'))
# coco_dt = coco_gt.loadRes(os.path.join(folder_path,'weight_adapted_samH_coco_instances_results_prediction.json'))


# for LMO
folder_path = '../datasets/lmo/test/000002'
coco_gt = COCO(os.path.join(folder_path, 'scene_gt_coco.json'))
coco_dt = coco_gt.loadRes(os.path.join(folder_path,'weight_samH_coco_instances_results.json'))

# Specify the directory containing the images
image_directory = folder_path

def create_color_map(num_classes, colormap_name):
    colormap = plt.get_cmap(colormap_name)
    num_colors = len(colormap.colors)
    colors = [colormap(i % num_colors) for i in range(num_classes)]
    return {i + 1: colors[i % len(colors)] for i in range(num_classes)}

# Generate a color map for 30 classes using the 'tab20b' colormap
color_map = create_color_map(21, 'Paired')

def visualize_image_with_classes(image_id, coco_gt, coco_dt, image_directory, color_map,save_img_name='test.png'):
    # Load the image
    image_info = coco_gt.loadImgs(image_id)[0]
    image_path = f"{image_directory}/{image_info['file_name']}"
    image = Image.open(image_path)

    fig, axs = plt.subplots(1, 2, figsize=(20, 10))

    # Function to add annotations
    def add_annotations(ax, anns, coco, color_map):
        for ann in anns:
            bbox = ann['bbox']
            if 'score' in ann:
                if ann["score"] < 0.5:
                    continue
                #ax.text(bbox[0], bbox[1] - 25, f'{ann["score"]:.2f}', color=color, fontsize=20)

            # Get the class ID and its name
            class_id = ann['category_id']
            color = color_map[class_id]
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=10, edgecolor=color,
                                     facecolor='none')
            ax.add_patch(rect)

            if 'score' in ann:
                ax.text(bbox[0], bbox[1] - 10, f'{class_id} {ann["score"]:.2f}', color=color, fontsize=22, weight='bold')  # , weight='bold'
                # ax.text(bbox[0]+10, bbox[1] - 25, f'{ann["score"]:.2f}', color=color, fontsize=20)
            else:
                ax.text(bbox[0], bbox[1] - 10, f'{class_id}', color=color, fontsize=22, weight='bold')

    # Ground Truth
    axs[0].imshow(image)
    # axs[0].set_title('Ground Truth')
    axs[0].axis('off')
    gt_annIds = coco_gt.getAnnIds(imgIds=image_info['id'], iscrowd=None)
    gt_anns = coco_gt.loadAnns(gt_annIds)
    add_annotations(axs[0], gt_anns, coco_gt, color_map)

    # Predictions
    axs[1].imshow(image)
    # axs[1].set_title('Prediction')
    axs[1].axis('off')
    dt_annIds = coco_dt.getAnnIds(imgIds=image_info['id'], iscrowd=None)
    dt_anns = coco_dt.loadAnns(dt_annIds)
    add_annotations(axs[1], dt_anns, coco_gt, color_map)  # Passing coco_gt for class names

    plt.tight_layout()
    # plt.show()
    plt.savefig('../result_images/lmo/'+ save_img_name, bbox_inches='tight', dpi=300)


# Visualize an example image
# Replace with an actual image ID from your dataset
image_id = 203
visualize_image_with_classes(image_id, coco_gt, coco_dt, image_directory, color_map, save_img_name='bold_test203.png')


# # Load COCO annotations
# with open('../datasets/RoboTools/test/000002/scene_gt_coco_ins.json') as f:
#     coco_annotations = json.load(f)
#
# # Assuming you're interested in a specific image, get its ID
# image_id_of_interest = 3  # Replace with your image ID of interest
#
# # Find the corresponding annotations
# annotations = [anno for anno in coco_annotations['annotations'] if anno['image_id'] == image_id_of_interest]
#
# # Load the image
# image_info = [img for img in coco_annotations['images'] if img['id'] == image_id_of_interest]
# image_path = '../datasets/RoboTools/test/000002/' + image_info['file_name']
# image = cv2.imread(image_path)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
#
# # Draw the bounding boxes on the image
# for anno in annotations:
#     bbox = anno['bbox']
#     x, y, width, height = bbox
#     cv2.rectangle(image, (int(x), int(y)), (int(x + width), int(y + height)), (0, 255, 0), 2)
#
# # Display the image with bounding boxes
# plt.figure(figsize=(10, 10))
# plt.imshow(image)
# plt.axis('off')
# plt.show()