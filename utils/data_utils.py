#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
These functions are work on a set of images in a directory.
"""
import cv2
import copy
import glob
import os
import re
import sys
import numpy as np
from PIL import Image
from subprocess import check_output


def minify(datadir, destdir, factors=[], resolutions=[], extend='png'):
    """Using mogrify to resize rgb image

    Args:
        datadir(str): source data path
        destdir(str): save path
        factor(int): ratio of original width or height
        resolutions(int): new width or height
    """
    imgs = [os.path.join(datadir, f) for f in sorted(os.listdir(datadir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]

    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(int(r))
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        if os.path.exists(destdir):
            continue

        print('Minifying', r, datadir)

        os.makedirs(destdir)
        check_output('cp {}/* {}'.format(datadir, destdir), shell=True)

        ext = imgs[0].split('.')[-1]
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', extend, '*.{}'.format(ext)])

        print(args)
        os.chdir(destdir)
        check_output(args, shell=True)
        os.chdir(wd)

        if ext != extend:
            check_output('rm {}/*.{}'.format(destdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')


def resizemask(datadir, destdir, factors=[], resolutions=[]):
    """Using PIL.Image.resize to resize binary images with nearest-neighbor

    Args:
        datadir(str): source data path
        destdir(str): save path
        factor(float): 1/N original width or height
        resolutions(int): new width or height
    """
    mask_paths = sorted([p for p in glob.glob(os.path.join(datadir, '*'))
                         if re.search('/*\.(jpg|jpeg|png|gif|bmp)', str(p))])
    old_size = np.array(Image.open(mask_paths[0])).shape
    if len(old_size) != 2:
        old_size = old_size[:2]

    for r in factors + resolutions:
        if isinstance(r, int):
            width = int(old_size[0] / r)
            height = int(old_size[1] / r)
        else:
            width = r[0]
            height = r[1]
        if os.path.exists(destdir):
            continue
        else:
            os.makedirs(destdir)

        for i, mask_path in enumerate(mask_paths):
            mask = Image.open(mask_path)
            new_mask = mask.resize((width, height))

            base_filename = mask_path.split('/')[-1]
            new_mask.save(os.path.join(destdir, base_filename))

        print('Done')


def getbbox(mask, exponent=1):
    """Computing bboxes of foreground in the masks

    Args:
        mask: binary image
        exponent(int): the size (width or height) should be a multiple of exponent
    """

    x_center = mask.shape[0] // 2
    y_center = mask.shape[1] // 2

    x, y = (mask != 0).nonzero()  # x:height; y:width
    bbox = [min(x), max(x), min(y), max(y)]

    # nearest rectangle box that height/width is the multipler of a factor
    x_min = np.max([bbox[1] - x_center, x_center - bbox[0]]) * 2
    y_min = np.max([bbox[3] - y_center, y_center - bbox[2]]) * 2
    new_x = int(np.ceil(x_min / exponent) * exponent)
    new_y = int(np.ceil(y_min / exponent) * exponent)
    # print("A rectangle to bound the object with width and height:", (new_y, new_x))

    bbox = [x_center - new_x // 2, x_center + new_x // 2,
            y_center - new_y // 2, y_center + new_y // 2]
    return bbox


def centercrop(img, new_size):
    """Computing bboxes of foreground in the masks

    Args:
        img: PIL image
        exponent(int): the size (width or height) should be a multiple of exponent
    """
    if len(new_size) == 2:
        new_width = new_size[0]
        new_height = new_size[1]
    else:
        print('ERROR: Valid size not found. Aborting')
        sys.exit()

    width, height = img.size
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = (width + new_width) // 2
    bottom = (height + new_height) // 2

    new_img = img.crop((left, top, right, bottom))

    return new_img


def invertmask(img, mask):
    # mask only has 0 and 1, extract the foreground
    fg = cv2.bitwise_and(img, img, mask=mask)

    # create white background
    black_bg = np.zeros(img.shape, np.uint8)
    white_bg = ~black_bg

    # masking the white background
    white_bg = cv2.bitwise_and(white_bg, white_bg, mask=mask)
    white_bg = ~white_bg

    # foreground will be added to the black area
    new_img = cv2.add(white_bg, img)

    # invert mask to 0 for foreground and 255 for background
    new_mask = np.where(mask == 0, 255, 0)

    return new_img, new_mask
    
def gen_square_crops(img, bbox):
    img_width, img_height = img.size
    x0, y0, x1, y1 = bbox

    # new square long side size
    new_size = max(x1 - x0, y1 - y0)

    # center coordinate
    center_x = x0 + (x1 - x0) // 2
    center_y = y0 + (y1 - y0) // 2

    # new bbox
    if center_x + new_size // 2 > img_width:
        new_x1 = copy.copy(img_width)
        new_x0 = new_x1 - new_size
    elif center_x - new_size // 2 < 0:
        new_x0 = 0
        new_x1 = copy.copy(new_size)
    else:
        new_x0 = center_x - new_size // 2
        new_x1 = new_x0 + new_size

    if center_y + new_size // 2 > img_height:
        new_y1 = copy.copy(img_height)
        new_y0 = new_y1 - new_size
    elif center_y - new_size // 2 < 0:
        new_y0 = 0
        new_y1 = copy.copy(new_size)
    else:
        new_y0 = center_y - new_size // 2
        new_y1 = new_y0 + new_size

    # pad 255s or 0s to make crops square
    new_img = Image.new('RGB', (new_size, new_size), (255, 255, 255))
    new_img.paste(img.crop((x0, y0, x1, y1)), (abs(x0 - new_x0), abs(y0 - new_y0)))

    return new_img
