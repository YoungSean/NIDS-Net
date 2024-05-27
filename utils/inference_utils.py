import torch
from torch import nn
import torchvision
import numpy as np
from absl import app, logging
from PIL import Image as PILImg
from .img_utils import masks_to_bboxes
from robokit.utils import annotate, overlay_masks
import cv2
import os
from PIL import Image
from .data_utils import gen_square_crops
import json
import matplotlib.pyplot as plt
import random
from tqdm import trange

from detectron2.structures import Boxes, BoxMode, Instances
from detectron2.layers.nms import batched_nms

# from dinov2.eval.utils import extract_features
from pycocotools import mask as maskUtils
import torch.nn.functional as F
import torchvision.transforms as T

def compute_similarity(obj_feats, roi_feats):
    """
    Compute Cosine similarity between object features and proposal features
    """
    roi_feats = roi_feats.unsqueeze(-2)
    sim = torch.nn.functional.cosine_similarity(roi_feats, obj_feats, dim=-1)
    return sim

def stableMatching(preferenceMat):
    """
    Compute Stable Matching
    """
    mDict = dict()

    engageMatrix = np.zeros_like(preferenceMat)
    for i in range(preferenceMat.shape[0]):
        tmp = preferenceMat[i]
        sortIndices = np.argsort(tmp)[::-1]
        mDict[i] = sortIndices.tolist()

    freeManList = list(range(preferenceMat.shape[0]))

    while freeManList:
        curMan = freeManList.pop(0)
        curWoman = mDict[curMan].pop(0)
        if engageMatrix[:, curWoman].sum() == 0:
            engageMatrix[curMan, curWoman] = 1
        else:
            engagedMan = np.where(engageMatrix[:, curWoman] == 1)[0][0]
            if preferenceMat[engagedMan, curWoman] > preferenceMat[curMan, curWoman]:
                freeManList.append(curMan)
            else:
                engageMatrix[engagedMan, curWoman] = 0
                engageMatrix[curMan, curWoman] = 1
                freeManList.append(engagedMan)
    return engageMatrix


def get_bbox_masks_from_gdino_sam(image_path, gdino, SAM, text_prompt='objects', visualize=False):
    """
    Get bounding boxes and masks from gdino and sam
    @param image_path: the image path
    @param gdino: the model of grounding dino
    @param SAM: segment anything model or its variants
    @param text_prompt: generally 'objects' for object detection of noval objects
    @param visualize: if True, visualize the result
    @return: the bounding boxes and masks of the objects.
    Bounding boxes are in the format of [x_min, y_min, x_max, y_max] and shape of (N, 4).
    Masks are in the format of (N, H, W) and the value is True for object and False for background.
    They are both in the format of torch.tensor.
    """
    # logging.info("Open the image and convert to RGB format")
    image_pil = PILImg.open(image_path).convert("RGB")

    logging.info("GDINO: Predict bounding boxes, phrases, and confidence scores")
    with torch.no_grad():
        bboxes, phrases, gdino_conf = gdino.predict(image_pil, text_prompt)

        # logging.info("GDINO post processing")
        w, h = image_pil.size  # Get image width and height
        # Scale bounding boxes to match the original image size
        image_pil_bboxes = gdino.bbox_to_scaled_xyxy(bboxes, w, h)

        logging.info("SAM prediction")
        image_pil_bboxes, masks = SAM.predict(image_pil, image_pil_bboxes)
    masks = masks.squeeze(1)
    accurate_bboxs = masks_to_bboxes(masks)  # get the accurate bounding boxes from the masks
    accurate_bboxs = torch.tensor(accurate_bboxs)
    if visualize:
        logging.info("Annotate the scaled image with bounding boxes, confidence scores, and labels, and display")
        bbox_annotated_pil = annotate(overlay_masks(image_pil, masks), accurate_bboxs, gdino_conf, phrases)
        bbox_annotated_pil.show()
    return accurate_bboxs, masks

def get_object_proposal(image_path, bboxs, masks, tag="mask", ratio=1.0, save_rois=True, output_dir='object_proposals', save_segm=False, save_proposal=False):
    """
    Get object proposals from the image according to the bounding boxes and masks.

    @param image_path:
    @param bboxs: numpy array, the bounding boxes of the objects [N, 4]
    @param masks: Boolean numpy array of shape [N, H, W], True for object and False for background
    @param tag: use mask or bbox to crop the object
    @param ratio: ratio to resize the image
    @param save_rois: if True, save the cropped object proposals
    @param output_dir: the folder to save the cropped object proposals
    @return: the cropped object proposals and the object proposals information
    """
    raw_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    image_height, image_width = raw_image.shape[:-1]
    scene_name = os.path.basename(image_path).split('.')[0]
    sel_rois = []
    rois = []
    cropped_masks = []
    cropped_imgs = []
    # ratio = 0.25
    if ratio != 1.0:
        scene_image = cv2.resize(raw_image, (int(raw_image.shape[1] * ratio), int(raw_image.shape[0] * ratio)),
                               cv2.INTER_LINEAR)
    else:
        scene_image = raw_image
    # scene_image = cv2.resize(raw_image, (int(raw_image.shape[1] * ratio), int(raw_image.shape[0] * ratio)),
    #                          cv2.INTER_LINEAR)
    for ind in range(len(masks)):
        # bbox
        x0 = int(bboxs[ind][0])
        y0 = int(bboxs[ind][1])
        x1 = int(bboxs[ind][2])
        y1 = int(bboxs[ind][3])

        # load mask
        mask = masks[ind].squeeze(0).cpu().numpy()
        # Assuming `mask` is your boolean numpy array with shape (H, W)
        rle = None
        if save_segm:
            rle = maskUtils.encode(np.asfortranarray(mask.astype(np.uint8)))
            rle['counts'] = rle['counts'].decode('ascii')  # If saving to JSON, ensure counts is a string
        cropped_mask = mask[y0:y1, x0:x1]
        cropped_mask = Image.fromarray(cropped_mask.astype(np.uint8) * 255)
        cropped_masks.append(cropped_mask)
        # show mask
        cropped_img = raw_image[y0:y1, x0:x1]
        cropped_img = Image.fromarray(cropped_img)
        # cropped_img.show()
        # cropped_mask.show()
        # try masked image
        # cropped_mask_array = np.array(cropped_mask).astype(bool)
        # cropped_masked_img = cropped_img * cropped_mask_array[:, :, None]
        # cropped_img = Image.fromarray(cropped_masked_img)

        cropped_imgs.append(cropped_img)

        # save roi region
        if save_rois:
            # invert background to white
            new_image = Image.new('RGB', size=(image_width, image_height), color=(255, 255, 255))
            new_image.paste(Image.fromarray(raw_image), (0, 0),
                            mask=Image.fromarray(mask).resize((image_width, image_height)))
            if tag == "mask":
                roi = gen_square_crops(new_image, [x0, y0, x1, y1])  # crop by mask
            elif tag == "bbox":
                roi = gen_square_crops(Image.fromarray(raw_image), [x0, y0, x1, y1])  # crop by bbox
            else:
                ValueError("Wrong tag!")

            rois.append(roi)
            os.makedirs(os.path.join(output_dir, scene_name), exist_ok=True)
            roi.save(os.path.join(output_dir, scene_name, scene_name + '_' + str(ind).zfill(3) + '.png'))

        # save bbox
        sel_roi = dict()
        sel_roi['roi_id'] = int(ind)
        sel_roi['image_id'] = int(scene_name.split('_')[-1])
        sel_roi['bbox'] = [int(x0 * ratio), int(y0 * ratio), int((x1 - x0) * ratio), int((y1 - y0) * ratio)]
        sel_roi['area'] = np.count_nonzero(mask)
        # if you need segmentation mask, uncomment the following line
        # sel_roi['mask'] = mask  # boolean numpy array. H X W
        sel_roi['roi_dir'] = os.path.join(output_dir, scene_name, scene_name + '_' + str(ind).zfill(3) + '.png')
        sel_roi['image_dir'] = image_path
        sel_roi['image_width'] = scene_image.shape[1]
        sel_roi['image_height'] = scene_image.shape[0]
        if save_segm:
            sel_roi['segmentation'] = rle  # Add RLE segmentation
        sel_roi['scale'] = int(1 / ratio)
        sel_rois.append(sel_roi)
    if save_proposal:
        with open(os.path.join(output_dir, 'proposals_on_' + scene_name + '.json'), 'w') as f:
            json.dump(sel_rois, f)
    return rois, sel_rois, cropped_imgs, cropped_masks

def crop_images_and_masks(images, masks, bboxes, img_size=224):
    """
    Crop images and masks according to the given bounding boxes.

    Parameters:
        images (torch.Tensor): Tensor of shape [batch_size, channels, height, width].
        masks (torch.Tensor): Tensor of shape [batch_size, height, width].
        bboxes (torch.Tensor): Tensor of shape [batch_size, 4] with each row [y1, x1, y2, x2].

    Returns:
        cropped_images (torch.Tensor): Tensor of cropped images.
        cropped_masks (torch.Tensor): Tensor of cropped masks.
    """
    cropped_images = []
    cropped_masks = []

    for image, mask, bbox in zip(images, masks, bboxes):
        x0, y0, x1, y1 = bbox
        if x0 == x1 or y0 == y1:
            continue
        cropped_image = image[:, y0:y1, x0:x1]
        cropped_mask = mask[:, y0:y1, x0:x1]
        if cropped_image.size(1) == 0 or cropped_image.size(2) == 0:
            continue
        cropped_image = F.interpolate(cropped_image.unsqueeze(0), size=(img_size, img_size), mode='bicubic')
        cropped_mask = F.interpolate(cropped_mask.unsqueeze(0), size=(img_size, img_size), mode='bicubic')

        cropped_images.append(cropped_image)
        cropped_masks.append(cropped_mask)

    # Stack the list of tensors into a single tensor
    cropped_images = torch.cat(cropped_images, dim=0)
    cropped_masks = torch.cat(cropped_masks, dim=0)

    return cropped_images, cropped_masks
def get_object_proposal_tensor(image_path, bboxs, masks, img_size=448,rgb_normalize=None, tag="mask", ratio=1.0, save_rois=True, output_dir='object_proposals', save_segm=False):
    """
    Get object proposals from the image according to the bounding boxes and masks.

    @param image_path:
    @param bboxs: numpy array, the bounding boxes of the objects [N, 4]
    @param masks: Boolean numpy array of shape [N, H, W], True for object and False for background
    @param tag: use mask or bbox to crop the object
    @param ratio: ratio to resize the image
    @param save_rois: if True, save the cropped object proposals
    @param output_dir: the folder to save the cropped object proposals
    @return: the cropped object proposals and the object proposals information
    """
    raw_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    image_height, image_width = raw_image.shape[:-1]
    scene_name = os.path.basename(image_path).split('.')[0]
    sel_rois = []
    rois = []
    cropped_masks = []
    cropped_imgs = []
    # ratio = 0.25
    if ratio != 1.0:
        scene_image = cv2.resize(raw_image, (int(raw_image.shape[1] * ratio), int(raw_image.shape[0] * ratio)),
                               cv2.INTER_LINEAR)
    else:
        scene_image = raw_image
    # scene_image = cv2.resize(raw_image, (int(raw_image.shape[1] * ratio), int(raw_image.shape[0] * ratio)),
    #                          cv2.INTER_LINEAR)
    rgb = rgb_normalize(raw_image).float()
    # rgbs = rgb.unsqueeze(0).repeat(len(masks), 1, 1, 1)
    for ind in range(len(masks)):
        # bbox
        x0 = int(bboxs[ind][0])
        y0 = int(bboxs[ind][1])
        x1 = int(bboxs[ind][2])
        y1 = int(bboxs[ind][3])
        mask = masks[ind]
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)
        # load mask
        if x0 == x1 or y0 == y1:
            continue
        cropped_img = rgb[:, y0:y1, x0:x1]
        cropped_mask = mask[:, y0:y1, x0:x1]
        if cropped_img.size(1) == 0 or cropped_img.size(2) == 0:
            continue
        cropped_img = F.interpolate(cropped_img.unsqueeze(0), size=(img_size, img_size), mode='bicubic')
        cropped_mask = F.interpolate(cropped_mask.unsqueeze(0), size=(img_size // 14, img_size // 14), mode='bicubic')

        cropped_imgs.append(cropped_img)
        cropped_masks.append(cropped_mask)

        # save bbox
        mask = masks[ind].squeeze(0).cpu().numpy()
        # Assuming `mask` is your boolean numpy array with shape (H, W)
        rle = None
        if save_segm:
            rle = maskUtils.encode(np.asfortranarray(mask.astype(np.uint8)))
            rle['counts'] = rle['counts'].decode('ascii')  # If saving to JSON, ensure counts is a string
        # save roi region
        if save_rois:
            # invert background to white
            new_image = Image.new('RGB', size=(image_width, image_height), color=(255, 255, 255))
            new_image.paste(Image.fromarray(raw_image), (0, 0),
                            mask=Image.fromarray((mask).astype(np.uint8)).resize((image_width, image_height)))
            if tag == "mask":
                roi = gen_square_crops(new_image, [x0, y0, x1, y1])  # crop by mask
            elif tag == "bbox":
                roi = gen_square_crops(Image.fromarray(raw_image), [x0, y0, x1, y1])  # crop by bbox
            else:
                ValueError("Wrong tag!")

            rois.append(roi)
            os.makedirs(os.path.join(output_dir, scene_name), exist_ok=True)
            roi.save(os.path.join(output_dir, scene_name, scene_name + '_' + str(ind).zfill(3) + '.png'))
        sel_roi = dict()
        sel_roi['roi_id'] = int(ind)
        sel_roi['image_id'] = int(scene_name.split('_')[-1])
        sel_roi['bbox'] = [int(x0 * ratio), int(y0 * ratio), int((x1 - x0) * ratio), int((y1 - y0) * ratio)]
        sel_roi['area'] = np.count_nonzero(mask)
        sel_roi['roi_dir'] = os.path.join(output_dir, scene_name, scene_name + '_' + str(ind).zfill(3) + '.png')
        sel_roi['image_dir'] = image_path
        sel_roi['image_width'] = scene_image.shape[1]
        sel_roi['image_height'] = scene_image.shape[0]
        if save_segm:
            sel_roi['segmentation'] = rle  # Add RLE segmentation
        sel_roi['scale'] = int(1 / ratio)
        sel_rois.append(sel_roi)

    # Stack the list of tensors into a single tensor
    cropped_imgs = torch.cat(cropped_imgs, dim=0)
    cropped_masks = torch.cat(cropped_masks, dim=0)
    with open(os.path.join(output_dir, 'proposals_on_' + scene_name + '.json'), 'w') as f:
        json.dump(sel_rois, f)
    return rois, sel_rois, cropped_imgs, cropped_masks

def show_anns(anns):
    """ref from segment-anything's notebook
    """
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))

def getColor():
    color: int
    c1 = random.randint(16, 255)
    c2 = random.randint(16, 255)
    c3 = random.randint(16, 255)
    return (c1, c2, c3)

def create_instances(predictions, image_size, metadata, scale=1, CHOSEN_THRESHOLD=0.4):
    ret = Instances(image_size)

    score = np.asarray([x["score"] for x in predictions])
    chosen = (score > CHOSEN_THRESHOLD).nonzero()[0]
    score = score[chosen]
    bbox = np.asarray([predictions[i]["bbox"] for i in chosen]).reshape(-1, 4) * scale
    bbox = BoxMode.convert(bbox, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)

    labels = np.asarray([metadata.thing_dataset_id_to_contiguous_id[predictions[i]["category_id"]] for i in chosen])

    ret.scores = score
    ret.pred_boxes = Boxes(bbox)
    ret.pred_classes = labels

    try:
        ret.pred_masks = [predictions[i]["segmentation"] for i in chosen]
    except KeyError:
        pass
    return ret

def nms(pred_boxes, pred_scores, pred_classes):
    # Boxes
    boxes = pred_boxes.clone()
    # Scores
    scores = pred_scores.clone()
    # Classes
    classes = pred_classes.clone()

    # Perform nms
    iou_threshold = 0.5
    keep_id = batched_nms(boxes, scores, classes, iou_threshold)

    return keep_id


def apply_nms(pred):
    pred_boxes = pred.pred_boxes
    pred_scores = pred.scores
    pred_classes = pred.pred_classes
    iou_threshold = 0.5

    boxes_tmp = []
    scores_tmp = []
    classes_tmp = []

    for i, coordinates in enumerate(pred_boxes):
        boxes_tmp.append(coordinates.cpu().numpy())  # boxes_tmp.append(coordinates.cpu().numpy())
        scores_tmp.append(pred_scores[i])  # scores_tmp.append(pred_scores[idx].cpu())
        classes_tmp.append(pred_classes[i])  # classes_tmp.append(pred_classes[idx].cpu())

    boxes_tmp = torch.tensor(boxes_tmp)
    scores_tmp = torch.tensor(scores_tmp)
    classes_tmp = torch.tensor(classes_tmp)
    if len(boxes_tmp) != 0:
        keep_ids = batched_nms(boxes_tmp.cuda(), scores_tmp.cuda(), classes_tmp.cuda() * 0 + 1, iou_threshold)
    else:
        print("no box prediction!")
    return keep_ids

# def get_scene_feature(output_dir, scene_name, scene_dataset, model, batch_size=1, num_workers=0):
#     """
#     Get scene features from dinov2 model.
#     If the scene features are already saved, load them from the file.
#     Otherwise, extract the scene features and save them to the file.
#     @param output_dir: the folder to save the scene features
#     @param scene_name: the scene image filename
#     @param scene_dataset: the dataset of object proposals from the scene
#     @param model: dinov2 model
#     @param batch_size: batch size during inference with dinov2 model
#     @param num_workers: num of workers during inference with dinov2 model
#     @return: normalized scene features
#     """
#     if os.path.exists(os.path.join(output_dir, 'scene_features_' + scene_name + '.json')):
#         with open(os.path.join(output_dir, 'scene_features_' + scene_name + '.json'), 'r') as f:
#             feat_dict = json.load(f)
#
#         scene_features = torch.Tensor(feat_dict['features']).cuda()
#
#     else:
#         scene_features, _ = extract_features(
#             model, scene_dataset, batch_size, num_workers
#         )
#         feat_dict = dict()
#         feat_dict['features'] = scene_features.detach().cpu().tolist()
#
#         with open(os.path.join(output_dir, 'scene_features_' + scene_name + '.json'), 'w') as f:
#             json.dump(feat_dict, f)
#
#     # normalize features
#     scene_features = nn.functional.normalize(scene_features, dim=1, p=2)
#     return scene_features

# def get_object_features(output_dir, object_dataset, model, batch_size, num_workers):
#     """
#     Get object features from dinov2 model.
#     If the object features are already saved, load them from the file.
#     Otherwise, extract the object features and save them to the file.
#     @param output_dir: the folder to save the object features
#     @param object_dataset: the dataset of object templates
#     @param model: dinov2 model
#     @param batch_size: batch size during inference with dinov2 model
#     @param num_workers: num of workers during inference with dinov2 model
#     @return: normalized object features
#     """
#     if os.path.exists(os.path.join(output_dir, 'object_features.json')):
#         with open(os.path.join(output_dir, 'object_features.json'), 'r') as f:
#             feat_dict = json.load(f)
#
#         object_features = torch.Tensor(feat_dict['features']).cuda()
#
#     else:
#         object_features, _ = extract_features(
#             model, object_dataset, batch_size, num_workers
#         )
#
#         feat_dict = dict()
#         feat_dict['features'] = object_features.detach().cpu().tolist()
#
#         with open(os.path.join(output_dir, 'object_features.json'), 'w') as f:
#             json.dump(feat_dict, f)
#
#     # normalize features
#     object_features = nn.functional.normalize(object_features, dim=1, p=2)
#
#     return object_features


def FFA_preprocess(x_list, img_size=336):

    preprocessed_images = []

    for x in x_list:
        # width, height = x.size
        new_width = img_size
        new_height = img_size

        def _to_rgb(x):
            if x.mode != "RGB":

                x = x.convert("RGB")
            return x

        preprocessed_image = torchvision.transforms.Compose([
            _to_rgb,
            torchvision.transforms.Resize((new_height, new_width), interpolation=Image.BICUBIC),  # Image.BICUBIC / InterpolationMode.BICUBIC
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])(x)
        preprocessed_images.append(preprocessed_image)
    return torch.stack(preprocessed_images, dim=0)

def get_foreground_mask(masks, mask_size=24):
    """
    masks: list of PIL.Image. Resize masks to 24 * 24 and convert to tensor.
    @param masks:
    @return: Resized masks
    """
    new_masks = []
    for mask in masks:
        resized_mask = mask.resize((mask_size, mask_size), Image.BILINEAR)
        resized_mask_numpy = np.array(resized_mask)
        resized_mask_numpy = resized_mask_numpy / 255.0
        tensor_mask = torch.from_numpy(resized_mask_numpy.astype(np.float32))
        tensor_mask[tensor_mask > 0.5] = 1.0
        tensor_mask = tensor_mask.unsqueeze(0).long() #.to(self.device)
        if tensor_mask.sum() == 0:
            tensor_mask = torch.ones_like(tensor_mask)
        new_masks.append(tensor_mask)
    return torch.stack(new_masks, dim=0)


def get_features(images, masks, encoder, variant="Crop-Feat", device="cuda", img_size=336):
    """Get Foreground feature average from the model

    Args:
        images: input images. a list of PIL.Image
        masks: input masks. a list of PIL.Image
        model: model to extract features

    Returns:
        features: extracted features. shape of [N, C]
    """
    with torch.no_grad():
        preprocessed_imgs = FFA_preprocess(images, img_size).to(device)
        mask_size = img_size // 14
        masks = get_foreground_mask(masks, mask_size).to(device)
        if variant == "Crop-Feat":
            emb = encoder.forward_features(preprocessed_imgs)
        elif variant == "Crop-Img":
            emb = encoder.forward_features(FFA_preprocess(images, img_size).to(device))
        else:
            raise ValueError("Invalid variant, only Crop-Feat and Crop-Img are supported.")

        grid = emb["x_norm_patchtokens"].view(len(images), mask_size, mask_size, -1)
        avg_feature = (grid * masks.permute(0, 2, 3, 1)).sum(dim=(1, 2)) / masks.sum(dim=(1, 2, 3)).unsqueeze(-1)

        return avg_feature

def get_cls_token(images, masks, encoder, device="cuda", img_size=448):
    """Get cls token from the model

    Args:
        images: input images. a list of PIL.Image
        masks: input masks. a list of PIL.Image
        model: model to extract features

    Returns:
        features: extracted features. shape of [N, C]
    """
    rgb_normalize = T.Compose(
        [   T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    mask_resize = T.Compose(
        [   T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
        ]
    )

    with torch.no_grad():
        num_proposals = len(images)
        imgs = []
        for i in range(num_proposals):
            image = images[i]
            img = rgb_normalize(image)
            mask = masks[i]
            mask = mask_resize(mask)
            mask = np.array(mask)
            mask = mask / 255.0
            tensor_mask = torch.from_numpy(mask)
            tensor_mask[tensor_mask > 0.5] = 1.0
            tensor_mask = tensor_mask.unsqueeze(0)
            masked_img = img * tensor_mask
            imgs.append(masked_img)
        masked_imgs = torch.stack(imgs).to(device).float()
        emb = encoder.forward_features(masked_imgs)
        return emb['x_norm_clstoken']

def get_features_via_batch_tensor(preprocessed_imgs, masks, encoder, device="cuda", img_size=336):
    """Get Foreground feature average from the model

    Args:
        images: input images. a list of PIL.Image
        masks: input masks. a list of PIL.Image
        model: model to extract features

    Returns:
        features: extracted features. shape of [N, C]
    """
    with torch.no_grad():
        preprocessed_imgs = preprocessed_imgs.to(device)
        mask_size = img_size // 14
        masks = masks.to(device)
        emb = encoder.forward_features(preprocessed_imgs)
        grid = emb["x_norm_patchtokens"].view(len(preprocessed_imgs), mask_size, mask_size, -1)
        avg_feature = (grid * masks.permute(0, 2, 3, 1)).sum(dim=(1, 2)) / masks.sum(dim=(1, 2, 3)).unsqueeze(-1)
        return avg_feature


def get_FFA_features_and_cls_token(images, masks, encoder, variant="Crop-Feat", device="cuda", img_size=336):
    """Get Foreground feature average and cls token from the model
    Note you should use masked rgb image since cls_token is sensitive to noise!!

    Args:
        images: input images. a list of PIL.Image
        masks: input masks. a list of PIL.Image
        model: model to extract features

    Returns:
        features: extracted features. shape of [N, C]
        cls token: [1, C]
    """
    with torch.no_grad():
        preprocessed_imgs = FFA_preprocess(images, img_size).to(device)
        mask_size = img_size // 14
        masks = get_foreground_mask(masks, mask_size).to(device)
        if variant == "Crop-Feat":
            emb = encoder.forward_features(preprocessed_imgs)
        elif variant == "Crop-Img":
            emb = encoder.forward_features(FFA_preprocess(images, img_size).to(device))
        else:
            raise ValueError("Invalid variant, only Crop-Feat and Crop-Img are supported.")

        grid = emb["x_norm_patchtokens"].view(len(images), mask_size, mask_size, -1)
        FFA_feature = (grid * masks.permute(0, 2, 3, 1)).sum(dim=(1, 2)) / masks.sum(dim=(1, 2, 3)).unsqueeze(-1)
        cls_token = emb["x_norm_clstoken"]
        return FFA_feature, cls_token

def get_object_masked_FFA_features(output_dir, object_dataset, model, img_size=448):
    if os.path.exists(os.path.join(output_dir, 'object_features.json')):
        with open(os.path.join(output_dir, 'object_features.json'), 'r') as f:
            feat_dict = json.load(f)

        object_features = torch.Tensor(feat_dict['features']).cuda()

    else:
        object_features = []
        for i in trange(len(object_dataset)):
            img, _, mask = object_dataset[i]
            mask = mask.convert('L')
            ffa_feature = get_features([img], [mask], model, img_size=img_size)
            object_features.append(ffa_feature)
        object_features = torch.cat(object_features, dim=0)

        feat_dict = dict()
        feat_dict['features'] = object_features.detach().cpu().tolist()

        with open(os.path.join(output_dir, 'object_features.json'), 'w') as f:
            json.dump(feat_dict, f)

    # normalize features
    object_features = nn.functional.normalize(object_features, dim=1, p=2)

    return object_features

def resize_and_pad(img, final_size=(448, 448), padding_color_image=(255, 255, 255), padding_color_mask=0):
    # Load the image or mask
    # img = Image.open(image_path)

    # Calculate the resize target maintaining aspect ratio
    aspect_ratio = img.width / img.height
    if aspect_ratio > 1:  # Width is greater than height
        new_width = final_size[0]
        new_height = round(new_width / aspect_ratio)
    else:
        new_height = final_size[1]
        new_width = round(new_height * aspect_ratio)

    # Resize the image or mask
    img_resized = img.resize((new_width, new_height), Image.ANTIALIAS)

    # Calculate padding sizes
    padding_left = (final_size[0] - new_width) // 2
    padding_top = (final_size[1] - new_height) // 2
    padding_right = final_size[0] - new_width - padding_left
    padding_bottom = final_size[1] - new_height - padding_top

    # Determine padding color (white for images, black for masks)
    if img.mode == "L":  # Grayscale, likely a mask
        padding_color = padding_color_mask
    else:
        padding_color = padding_color_image

    # Create a new image with the specified dimensions and padding color
    img_padded = Image.new(img.mode, final_size, padding_color)
    img_padded.paste(img_resized, (padding_left, padding_top))

    # show the padded image
    # img.show()
    # img_padded.show()
    return img_padded

def get_weighted_FFA_features(images, masks, encoder, weightCNN, device="cuda", img_size=448):
    """Get Foreground feature average from the model

    Args:
        images: input images. a list of PIL.Image
        masks: input masks. a list of PIL.Image
        model: model to extract features

    Returns:
        features: extracted features. shape of [N, C]
    """
    with torch.no_grad():
        preprocessed_imgs = FFA_preprocess(images, img_size).to(device)
        mask_size = img_size // 14
        masks = get_foreground_mask(masks, mask_size).to(device)

        emb = encoder.forward_features(preprocessed_imgs)

        grid = emb["x_norm_patchtokens"].view(len(images), mask_size, mask_size, -1)
        grid = grid.permute(0, 3, 1, 2)
        avg_feature = weightCNN(grid, masks)
        return avg_feature

