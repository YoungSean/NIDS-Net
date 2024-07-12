#!/usr/bin/env python

# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

"""Test UCN on ros images"""

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
import _init_paths
# import networks
import rospy
import ros_numpy
import copy
import scipy.io

from utils.blob import pad_im
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from fcn.config import cfg, cfg_from_file, get_output_dir
from fcn.test_dataset import test_sample
from utils.mask import visualize_segmentation
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

        # bgr image
        im = im_color.astype(np.float32)
        im_tensor = torch.from_numpy(im) / 255.0
        pixel_mean = torch.tensor(cfg.PIXEL_MEANS / 255.0).float()
        im_tensor -= pixel_mean
        image_blob = im_tensor.permute(2, 0, 1)
        sample = {'image_color': image_blob.unsqueeze(0)}

        if cfg.INPUT == 'DEPTH' or cfg.INPUT == 'RGBD':
            height = im_color.shape[0]
            width = im_color.shape[1]
            depth_img[np.isnan(depth_img)] = 0
            xyz_img = compute_xyz(depth_img, self.fx, self.fy, self.px, self.py, height, width)
            depth_blob = torch.from_numpy(xyz_img).permute(2, 0, 1)
            sample['depth'] = depth_blob.unsqueeze(0)

        # out_label, out_label_refined = test_sample(sample, self.network, self.network_crop)
        # input imaeg: RGB numpy array.
        im_rgb = cv2.cvtColor(im_color, cv2.COLOR_BGR2RGB)
        results, mask = self.model.step(im_rgb)
        print(results)
        print("all masks: ", mask)

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

        # if out_label_refined is not None:
        #     label_refined = out_label_refined[0].cpu().numpy()
        #     label_msg_refined = self.cv_bridge.cv2_to_imgmsg(label_refined.astype(np.uint8))
        #     label_msg_refined.header.stamp = rgb_frame_stamp
        #     label_msg_refined.header.frame_id = rgb_frame_id
        #     label_msg_refined.encoding = 'mono8'
        #     self.label_refined_pub.publish(label_msg_refined)

        # publish segmentation images
        im_label = visualize_segmentation(im_color[:, :, (2, 1, 0)], label, return_rgb=True)
        rgb_msg = ros_numpy.msgify(Image, im_label, 'rgb8')
        rgb_msg.header.stamp = rgb_frame_stamp
        rgb_msg.header.frame_id = rgb_frame_id
        self.image_pub.publish(rgb_msg)

        # if out_label_refined is not None:
        #     im_label_refined = visualize_segmentation(im_color[:, :, (2, 1, 0)], label_refined, return_rgb=True)
        #     rgb_msg_refined = self.cv_bridge.cv2_to_imgmsg(im_label_refined, 'rgb8')
        #     rgb_msg_refined.header.stamp = rgb_frame_stamp
        #     rgb_msg_refined.header.frame_id = rgb_frame_id
        #     self.image_refined_pub.publish(rgb_msg_refined)
            
        # save results
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
    # parser.add_argument('--instance', dest='instance_id', help='PoseCNN instance id to use',
    #                     default=0, type=int)
    # parser.add_argument('--pretrained', dest='pretrained',
    #                     help='initialize with pretrained checkpoint',
    #                     default=None, type=str)
    # parser.add_argument('--pretrained_crop', dest='pretrained_crop',
    #                     help='initialize with pretrained checkpoint for crops',
    #                     default=None, type=str)
    # parser.add_argument('--cfg', dest='cfg_file',
    #                     help='optional config file', default=None, type=str)
    # parser.add_argument('--dataset', dest='dataset_name',
    #                     help='dataset to train on',
    #                     default='shapenet_scene_train', type=str)
    # parser.add_argument('--rand', dest='randomize',
    #                     help='randomize (do not use a fixed seed)',
    #                     action='store_true')
    # parser.add_argument('--network', dest='network_name',
    #                     help='name of the network',
    #                     default=None, type=str)
    # parser.add_argument('--background', dest='background_name',
    #                     help='name of the background file',
    #                     default=None, type=str)

    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    # if args.cfg_file is not None:
    #     cfg_from_file(args.cfg_file)

    print('Using config:')
    pprint.pprint(cfg)

    # if not args.randomize:
    #     # fix the random seeds (numpy and caffe) for reproducibility
    #     np.random.seed(cfg.RNG_SEED)

    # device
    # cfg.gpu_id = 0
    # cfg.device = torch.device('cuda:{:d}'.format(cfg.gpu_id))
    # cfg.instance_id = args.instance_id
    # num_classes = 2
    # cfg.MODE = 'TEST'
    # cfg.TEST.VISUALIZE = False
    # print('GPU device {:d}'.format(args.gpu_id))

    # prepare network
    # if args.pretrained:
    #     network_data = torch.load(args.pretrained)
    #     print("=> using pre-trained network '{}'".format(args.pretrained))
    # else:
    #     network_data = None
    #     print("no pretrained network specified")
    #     sys.exit()

    adapter_descriptors_path = "ros/weight_obj_shuffle2_0501_bs32_epoch_500_adapter_descriptors_pbr.json"
    with open(os.path.join(adapter_descriptors_path), 'r') as f:
        feat_dict = json.load(f)

    object_features = torch.Tensor(feat_dict['features']).cuda()
    object_features = object_features.view(-1, 42, 1024)
    weight_adapter_path = "ros/bop_obj_shuffle_weight_0430_temp_0.05_epoch_500_lr_0.001_bs_32_weights.pth"
    model = NIDS(object_features, use_adapter=True, adapter_path=weight_adapter_path)
    

    # image listener
    listener = ImageListener(model)
    print("is it good?")
    while not rospy.is_shutdown():
       listener.run_network()
