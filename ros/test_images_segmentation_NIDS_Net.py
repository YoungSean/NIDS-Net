#!/usr/bin/env python

# adapted from https://github.com/YoungSean/UnseenObjectsWithMeanShift/blob/master/ros/test_images_segmentation_transformer.py
"""Test NIDS-Net on ros images"""

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import tf
import rosnode
import message_filters
import cv2
import torch.nn as nn
import threading
import argparse
import pprint
import time, os, sys
import os.path as osp
import numpy as np
# import networks
import rospy
import ros_numpy
import copy
import scipy.io

from utils.blob import pad_im
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from utils.mask import visualize_results
lock = threading.Lock()

from nids_net import NIDS
import json


def compute_xyz(depth_img, fx, fy, px, py, height, width):
    indices = np.indices((height, width), dtype=np.float32).transpose(1,2,0)
    z_e = depth_img
    x_e = (indices[..., 1] - px) * z_e / fx
    y_e = (indices[..., 0] - py) * z_e / fy
    xyz_img = np.stack([x_e, y_e, z_e], axis=-1) # Shape: [H x W x 3]
    return xyz_img


class ImageListener:

    def __init__(self, model):

        self.model = model
        
        self.cv_bridge = CvBridge()

        self.im = None
        self.depth = None
        self.rgb_frame_id = None
        self.rgb_frame_stamp = None
        self.counter = 0
        self.output_dir = 'output/real_world'

        # initialize a node
        rospy.init_node("seg_rgb")
        self.label_pub = rospy.Publisher('seg_label', Image, queue_size=10)
        # self.label_refined_pub = rospy.Publisher('seg_label_refined', Image, queue_size=10)
        self.image_pub = rospy.Publisher('seg_image', Image, queue_size=10)
        # self.image_refined_pub = rospy.Publisher('seg_image_refined', Image, queue_size=10)
        self.feature_pub = rospy.Publisher('seg_feature', Image, queue_size=10)

        self.base_frame = 'base_link'
        rgb_sub = message_filters.Subscriber('/head_camera/rgb/image_raw', Image, queue_size=10)
        depth_sub = message_filters.Subscriber('/head_camera/depth_registered/image_raw', Image, queue_size=10)
        msg = rospy.wait_for_message('/head_camera/rgb/camera_info', CameraInfo)
        self.camera_frame = 'head_camera_rgb_optical_frame'
        self.target_frame = self.base_frame

        # update camera intrinsics
        intrinsics = np.array(msg.K).reshape(3, 3)
        self.fx = intrinsics[0, 0]
        self.fy = intrinsics[1, 1]
        self.px = intrinsics[0, 2]
        self.py = intrinsics[1, 2]
        print(intrinsics)

        queue_size = 1
        slop_seconds = 0.1
        ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size, slop_seconds)
        ts.registerCallback(self.callback_rgbd)


    def callback_rgbd(self, rgb, depth):

        if depth.encoding == '32FC1':
            depth_cv = ros_numpy.numpify(depth)
        elif depth.encoding == '16UC1':
            depth_cv = ros_numpy.numpify(depth).copy().astype(np.float32)
            depth_cv /= 1000.0
        else:
            rospy.logerr_throttle(
                1, 'Unsupported depth type. Expected 16UC1 or 32FC1, got {}'.format(
                    depth.encoding))
            return

        im = ros_numpy.numpify(rgb)

        with lock:
            self.im = im.copy()
            self.depth = depth_cv.copy()
            self.rgb_frame_id = rgb.header.frame_id
            self.rgb_frame_stamp = rgb.header.stamp


    def run_network(self):

        with lock:
            if listener.im is None:
              return
            im_color = self.im.copy()
            depth_img = self.depth.copy()
            rgb_frame_id = self.rgb_frame_id
            rgb_frame_stamp = self.rgb_frame_stamp

        print('===========================================')


        # input imaeg: RGB numpy array.
        im_rgb = cv2.cvtColor(im_color, cv2.COLOR_BGR2RGB)
        results, mask = self.model.step(im_rgb)
        #print(results)
        #print("all masks: ", mask)

        # publish segmentation mask
        # label = out_label[0].cpu().numpy()
        label = mask.cpu().numpy()
        label_msg = ros_numpy.msgify(Image, label.astype(np.uint8), 'mono8')
        label_msg.header.stamp = rgb_frame_stamp
        label_msg.header.frame_id = rgb_frame_id
        label_msg.encoding = 'mono8'
        self.label_pub.publish(label_msg)

        num_object = len(np.unique(label)) - 1
        print('%d objects' % (num_object))

        # publish segmentation images
        class_names = ["background", "002_master_chef_can", "003_cracker_box", "004_sugar_box", 
        "005_tomato_soup_can", "006_mustard_bottle", "007_tuna_fish_can", "008_pudding_box", "009_gelatin_box",
         "010_potted_meat_can", "011_banana", "019_pitcher_base", "021_bleach_cleanser", 
         "024_bowl", "025_mug", "035_power_drill", "036_wood_block", "037_scissors", 
         "040_large_marker", "051_large_clamp","052_extra_large_clamp","061_foam_brick"]
        im_label = visualize_results(im_color, results, class_names, return_rgb=True)
        rgb_msg = ros_numpy.msgify(Image, im_label, 'rgb8')
        rgb_msg.header.stamp = rgb_frame_stamp
        rgb_msg.header.frame_id = rgb_frame_id
        self.image_pub.publish(rgb_msg)
            
        # save results
        # to do. need to modify the following code
        save_result = False
        if save_result:
            # result = {'rgb': im_color, 'labels': label, 'labels_refined': label_refined}
            result = {'rgb': im_color, 'labels': label}
            filename = os.path.join(self.output_dir, '%06d.mat' % self.counter)
            print(filename)
            scipy.io.savemat(filename, result, do_compression=True)
            filename = os.path.join(self.output_dir, '%06d.jpg' % self.counter)
            cv2.imwrite(filename, im_color)
            filename = os.path.join(self.output_dir, '%06d-label.jpg' % self.counter)
            cv2.imwrite(filename, im_label[:, :, (2, 1, 0)])
            # filename = os.path.join(self.output_dir, '%06d-label-refined.jpg' % self.counter)
            # cv2.imwrite(filename, im_label_refined[:, :, (2, 1, 0)])
            self.counter += 1
            sys.exit(1)


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a NIDS-Net network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)


    
    adapter_descriptors_path = "ros/weight_obj_shuffle2_0501_bs32_epoch_500_adapter_descriptors_pbr.json"
    with open(os.path.join(adapter_descriptors_path), 'r') as f:
        feat_dict = json.load(f)

    object_features = torch.Tensor(feat_dict['features']).cuda()
    object_features = object_features.view(-1, 42, 1024)
    weight_adapter_path = "ros/bop_obj_shuffle_weight_0430_temp_0.05_epoch_500_lr_0.001_bs_32_weights.pth"
    model = NIDS(object_features, use_adapter=True, adapter_path=weight_adapter_path)
    

    # image listener
    listener = ImageListener(model)
    while not rospy.is_shutdown():
       listener.run_network()
