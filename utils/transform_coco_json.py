import json

# Step 1: Load the JSON file
# lmo1 = json.load(open('../datasets/lmo/test/000002/scene_gt_coco.json'))

def transform_json(scene_id, offset=0):
    with open(f'../datasets/RoboTools/test/{str(scene_id).zfill(6)}/scene_gt_coco_ins.json', 'r') as file:
        data = json.load(file)

    images = data['images'][offset:]
    annotations = data['annotations'][offset:]
    new_offset = len(data['images'])
    # from id to its img id
    new_images = []
    old2new = {}
    new_ids = set()
    for i, img in enumerate(images):
        new_imgid = int(img['file_name'].split('/')[-1].split('.')[0])
        img['id'] = new_imgid
        old2new[i] = new_imgid
        if new_imgid in new_ids:
            continue
        else:
            new_ids.add(new_imgid)
            new_images.append(img)

    new_annotations = []
    for ann in annotations:
        ann['image_id'] = old2new[ann['image_id']]
        new_annotations.append(ann)

    data['images'] = new_images
    data['annotations'] = new_annotations

    with open(f'../datasets/RoboTools/test/{str(scene_id).zfill(6)}/current_scene_gt_coco.json', 'w') as file:
        json.dump(data, file)
    return new_offset


offset = 0
# for i in range(1, 25):
#     offset = transform_json(i, offset)
#     print(offset)
transform_json(1, 0)