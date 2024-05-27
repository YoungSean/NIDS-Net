import os
import random
import logging
import torch
import numpy as np
from typing import List
import supervision as sv
from PIL import Image as PILImg, ImageDraw


def file_exists(file_path):
    """
    Check if a file exists.

    Parameters:
    - file_path (str): Path to the file.

    Returns:
    - bool: True if the file exists, False otherwise.
    """
    return os.path.exists(file_path)


def crop_images(original_image, bounding_boxes):
    """
    Crop the input image using the provided bounding boxes.

    Parameters:
    - original_image (PIL.Image.Image): Original input image.
    - bounding_boxes (list): List of bounding boxes [x_min, y_min, x_max, y_max].

    Returns:
    - cropped_images (list): List of cropped images.

    Raises:
    - ValueError: If the bounding box dimensions are invalid.
    """
    cropped_images = []

    try:
        for box in bounding_boxes:
            if len(box) != 4:
                raise ValueError("Bounding box should have 4 values: [x_min, y_min, x_max, y_max]")

            x_min, y_min, x_max, y_max = box

            # Check if the bounding box dimensions are valid
            if x_min < 0 or y_min < 0 or x_max <= x_min or y_max <= y_min:
                raise ValueError("Invalid bounding box dimensions")

            # Crop the image using the bounding box
            cropped_image = original_image.crop((x_min, y_min, x_max, y_max))
            cropped_images.append(cropped_image)

    except ValueError as e:
        print(f"Error in crop_images: {e}")


def annotate(image_source, boxes, logits, phrases):
    """
    Annotate image with bounding boxes, logits, and phrases.

    Parameters:
    - image_source (PIL.Image.Image): Input image source.
    - boxes (torch.tensor): Bounding boxes in xyxy format.
    - logits (list): List of confidence logits.
    - phrases (list): List of phrases.

    Returns:
    - PIL.Image: Annotated image.
    """
    try:
        detections = sv.Detections(xyxy=boxes.cpu().numpy())
        labels = [
            f"{phrase} {logit:.2f}"
            for phrase, logit
            in zip(phrases, logits)
        ]
        box_annotator = sv.BoxAnnotator(thickness=2)
        img_pil = PILImg.fromarray(box_annotator.annotate(scene=np.array(image_source), detections=detections, labels=labels))
        return img_pil
    
    except Exception as e:
        logging.error(f"Error during annotation: {e}")
        raise e

def draw_mask(mask, draw, random_color=False):
    """
    Draw a segmentation mask on an image.

    Parameters:
    - mask (numpy.ndarray): The segmentation mask as a NumPy array.
    - draw (PIL.ImageDraw.ImageDraw): The PIL ImageDraw object to draw on.
    - random_color (bool, optional): Whether to use a random color for the mask. Default is False.

    Returns:
    - None
    """
    # Define the color for the mask
    if random_color:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 153)
    else:
        color = (30, 144, 255, 153)

    # Get the coordinates of non-zero elements in the mask
    nonzero_coords = np.transpose(np.nonzero(mask))

    # Draw each non-zero coordinate on the image
    for coord in nonzero_coords:
        draw.point(coord[::-1], fill=color)


def overlay_masks(image_pil: PILImg, masks):
    """
    Overlay segmentation masks on the input image.

    Parameters:
    - image_pil (PIL.Image): The input image as a PIL image.
    - masks (List[Tensor]): List of segmentation masks as torch Tensors.

    Returns:
    - PIL.Image: The image with overlayed segmentation masks.
    """
    mask_image = PILImg.new('RGBA', image_pil.size, color=(0, 0, 0, 0))
    mask_draw = ImageDraw.Draw(mask_image)
    for mask in masks:
        draw_mask(mask[0].cpu().numpy(), mask_draw, random_color=True)

    image_pil = image_pil.convert('RGBA')
    image_pil.alpha_composite(mask_image)
    return image_pil.convert('RGB')


def combine_masks(gt_masks):
    """
    Combine several bit masks [N, H, W] into a mask [H,W],
    e.g. 8*480*640 tensor becomes a numpy array of 480*640.
    [[1,0,0], [0,1,0]] = > [1,2,0].
    """
    gt_masks = torch.flip(gt_masks, dims=(0,))
    num, h, w = gt_masks.shape
    # bin_mask = np.zeros((h, w))
    bin_mask = torch.zeros((h,w), device=gt_masks.device)
    num_instance = len(gt_masks)
    # if there is not any instance, just return a mask full of 0s.
    if num_instance == 0:
        return bin_mask

    for m, object_label in zip(gt_masks, range(1, 1+num_instance)):
        label_pos = torch.nonzero(m, as_tuple=True)
        bin_mask[label_pos] = object_label
    return bin_mask


def filter_large_boxes(boxes, w, h, threshold=0.5):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = (x2 - x1) * (y2 - y1)
    index = area < (w * h) * threshold
    return boxes[index], index.cpu()
