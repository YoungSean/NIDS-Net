# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import os
import os.path as osp
import numpy as np


class imdb(object):
    """Image database."""

    def __init__(self):
        self._name = ''
        self._num_classes = 0
        self._classes = []
        self._class_colors = []

    @property
    def name(self):
        return self._name

    @property
    def num_classes(self):
        return len(self._classes)

    @property
    def classes(self):
        return self._classes

    @property
    def class_colors(self):
        return self._class_colors

    @property
    def cache_path(self):
        cache_path = osp.abspath(osp.join(ROOT_DIR, 'data', 'cache'))
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        return cache_path


    # backproject pixels into 3D points in camera's coordinate system
    def backproject(self, depth_cv, intrinsic_matrix, factor):

        depth = depth_cv.astype(np.float32, copy=True) / factor

        index = np.where(~np.isfinite(depth))
        depth[index[0], index[1]] = 0

        # get intrinsic matrix
        K = intrinsic_matrix
        Kinv = np.linalg.inv(K)

        # compute the 3D points
        width = depth.shape[1]
        height = depth.shape[0]

        # construct the 2D points matrix
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        ones = np.ones((height, width), dtype=np.float32)
        x2d = np.stack((x, y, ones), axis=2).reshape(width*height, 3)

        # backprojection
        R = np.dot(Kinv, x2d.transpose())

        # compute the 3D points
        X = np.multiply(np.tile(depth.reshape(1, width*height), (3, 1)), R)
        return np.array(X).transpose().reshape((height, width, 3))