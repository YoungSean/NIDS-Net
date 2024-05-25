import json


# Load the full COCO ground truth JSON file
with open('datasets/lmo/test/000002/scene_gt_coco.json') as f:
    coco_data = json.load(f)

# Your specific image ID
image_id = 3

# Extract annotations for the specific image
image_annotations = [anno for anno in coco_data['annotations'] if anno['image_id'] == image_id]

# Optionally, you might also want to extract the image information
image_info = [img for img in coco_data['images'] if img['id'] == image_id][0]

print(image_info)
print(image_annotations)

# Prepare a new JSON structure for the single image
single_image_ground_truth = {
    "images": [image_info],
    "annotations": image_annotations,
    "categories": coco_data['categories']  # Assuming you want to keep all category info
}

# Save this as a new JSON file
with open('test_data/lmo_test/single_image_ground_truth_'+str(image_id)+'.json', 'w') as f:
    json.dump(single_image_ground_truth, f)

# # Save this as a new JSON file
# with open('test_data/lmo_test/single_image_ground_truth_003.json', 'w') as f:
#     json.dump(single_image_ground_truth, f)