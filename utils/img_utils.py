import torch
import numpy as np
from PIL import Image

def mask_to_bbox(mask):
    rows, cols = torch.where(mask)
    if rows.size(0) == 0:
        return None  # or (0, 0, 0, 0) as a placeholder for no bounding box
    x1 = torch.min(cols).item()
    y1 = torch.min(rows).item()
    x2 = torch.max(cols).item()
    y2 = torch.max(rows).item()
    # top_left = [torch.min(rows).item(), torch.min(cols).item()]
    # bottom_right = [torch.max(rows).item(), torch.max(cols).item()]
    return [x1, y1, x2, y2]


def masks_to_bboxes(masks):
    assert masks.dim() == 3, "Input must be a 3D tensor."
    assert masks.dtype == torch.bool, "Input must be a boolean tensor."

    bboxes = []
    for i in range(masks.size(0)):
        bbox = mask_to_bbox(masks[i])
        bboxes.append(bbox)
    return bboxes

def get_masked_image(img, mask):
    mask_array = np.array(mask)
    binary_mask = (mask_array > 0).astype(np.uint8)

    # Apply the mask to each channel of the image
    img_array = np.array(img)
    # get masked part, non-mask part is white
    masked_img_array = np.where(np.stack([binary_mask] * 3, axis=-1), img_array, 255 * np.ones_like(img_array))
    # masked_img_array = img_array * np.stack([binary_mask]*3, axis=-1)  # Stack the mask across the channel dimension

    # Convert the masked image back to a PIL Image
    masked_img = Image.fromarray(masked_img_array)
    return masked_img

def find_mask_bbox(mask_array):
    rows = np.any(mask_array, axis=1)
    cols = np.any(mask_array, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return (cmin, rmin, cmax+1, rmax+1)  # PIL uses (left, upper, right, lower)