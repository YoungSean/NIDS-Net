#!/usr/bin/env python

# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

"""Test MSMTransformer on ros images"""

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
import networks
import rospy
import ros_numpy
import copy
import scipy.io

from utils.blob import pad_im
from sensor_msgs.msg import Image, CameraInfo
from fcn.config import cfg, cfg_from_file, get_output_dir
from fcn.test_dataset import test_sample
from fcn.test_demo import get_predictor, get_predictor_crop
from fcn.test_utils import test_sample_crop_nolabel
from utils.mask import visualize_segmentation
lock = threading.Lock()


def compute_xyz(depth_img, fx, fy, px, py, height, width):
    indices = np.indices((height, width), dtype=np.float32).transpose(1,2,0)
    z_e = depth_img
    x_e = (indices[..., 1] - px) * z_e / fx
    y_e = (indices[..., 0] - py) * z_e / fy
    xyz_img = np.stack([x_e, y_e, z_e], axis=-1) # Shape: [H x W x 3]
    return xyz_img


class ImageListener:

    def __init__(self, predictor, predictor_crop, cfg_transformer, cfg_transformer_crop):

        self.predictor = predictor
        self.predictor_crop = predictor_crop
        self.cfg_transformer = cfg_transformer
        self.cfg_transformer_crop = cfg_transformer_crop

        self.im = None
        self.depth = None
        self.rgb_frame_id = None
        self.rgb_frame_stamp = None
        self.counter = 0
        self.output_dir = 'output/real_world'

        # initialize a node
        rospy.init_node("seg_rgb")
        self.label_pub = rospy.Publisher('seg_label', Image, queue_size=10)
        self.score_pub = rospy.Publisher('seg_score', Image, queue_size=10)        
        self.label_refined_pub = rospy.Publisher('seg_label_refined', Image, queue_size=10)
        self.image_pub = rospy.Publisher('seg_image', Image, queue_size=10)
        self.image_refined_pub = rospy.Publisher('seg_image_refined', Image, queue_size=10)

        if cfg.TEST.ROS_CAMERA  == 'Fetch':
            self.base_frame = 'base_link'
            rgb_sub = message_filters.Subscriber('/head_camera/rgb/image_raw', Image, queue_size=10)
            depth_sub = message_filters.Subscriber('/head_camera/depth_registered/image_raw', Image, queue_size=10)
            msg = rospy.wait_for_message('/head_camera/rgb/camera_info', CameraInfo)
            self.camera_frame = 'head_camera_rgb_optical_frame'
            self.target_frame = self.base_frame
        elif cfg.TEST.ROS_CAMERA == 'Realsense':
            # use RealSense D435
            self.base_frame = 'measured/base_link'
            rgb_sub = message_filters.Subscriber('/camera/color/image_raw', Image, queue_size=10)
            depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, queue_size=10)
            msg = rospy.wait_for_message('/camera/color/camera_info', CameraInfo)
            self.camera_frame = 'measured/camera_color_optical_frame'
            self.target_frame = self.base_frame
        elif cfg.TEST.ROS_CAMERA == 'Azure':
            self.base_frame = 'measured/base_link'
            rgb_sub = message_filters.Subscriber('/k4a/rgb/image_raw', Image, queue_size=10)
            depth_sub = message_filters.Subscriber('/k4a/depth_to_rgb/image_raw', Image, queue_size=10)
            msg = rospy.wait_for_message('/k4a/rgb/camera_info', CameraInfo)
            self.camera_frame = 'rgb_camera_link'
            self.target_frame = self.base_frame
        else:
            # use kinect
            self.base_frame = '%s_rgb_optical_frame' % (cfg.TEST.ROS_CAMERA)
            rgb_sub = message_filters.Subscriber('/%s/rgb/image_color' % (cfg.TEST.ROS_CAMERA), Image, queue_size=10)
            depth_sub = message_filters.Subscriber('/%s/depth_registered/image' % (cfg.TEST.ROS_CAMERA), Image, queue_size=10)
            msg = rospy.wait_for_message('/%s/rgb/camera_info' % (cfg.TEST.ROS_CAMERA), CameraInfo)
            self.camera_frame = '%s_rgb_optical_frame' % (cfg.TEST.ROS_CAMERA)
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

        # rescale image if necessary
        if cfg.TEST.SCALES_BASE[0] != 1:
            im_scale = cfg.TEST.SCALES_BASE[0]
            im = pad_im(cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR), 16)
            depth_cv = pad_im(cv2.resize(depth_cv, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_NEAREST), 16)

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
        sample = {'image_color': image_blob}

        if cfg.INPUT == 'DEPTH' or cfg.INPUT == 'RGBD':
            height = im_color.shape[0]
            width = im_color.shape[1]
            depth_img[np.isnan(depth_img)] = 0
            xyz_img = compute_xyz(depth_img, self.fx, self.fy, self.px, self.py, height, width)
            depth_blob = torch.from_numpy(xyz_img).permute(2, 0, 1)
            sample['depth'] = depth_blob

        # run the network
        # out_label, out_label_refined = test_sample(sample, self.network, self.network_crop)
        out_label, out_label_refined, out_score, bbox = test_sample_crop_nolabel(self.cfg_transformer, sample, self.predictor, self.predictor_crop, visualization=False, topk=False, confident_score=0.2, print_result=True)

        # publish segmentation mask
        label = out_label[0].cpu().numpy()
        label_msg = ros_numpy.msgify(Image, label.astype(np.uint8), 'mono8')
        label_msg.header.stamp = rgb_frame_stamp
        label_msg.header.frame_id = rgb_frame_id
        label_msg.encoding = 'mono8'
        self.label_pub.publish(label_msg)
        
        # publish score map
        score = out_score[0].cpu().numpy()
        label_msg = ros_numpy.msgify(Image, score.astype(np.uint8), 'mono8')
        label_msg.header.stamp = rgb_frame_stamp
        label_msg.header.frame_id = rgb_frame_id
        label_msg.encoding = 'mono8'
        self.score_pub.publish(label_msg)

        num_object = len(np.unique(label)) - 1
        print('%d objects' % (num_object))

        if out_label_refined is not None:
            label_refined = out_label_refined
            label_msg_refined = ros_numpy.msgify(Image, label_refined.astype(np.uint8), 'mono8')
            label_msg_refined.header.stamp = rgb_frame_stamp
            label_msg_refined.header.frame_id = rgb_frame_id
            label_msg_refined.encoding = 'mono8'
            self.label_refined_pub.publish(label_msg_refined)
            num_object = len(np.unique(label_refined)) - 1
            print('%d objects after refinement' % (num_object))

        # publish segmentation images
        im_label = visualize_segmentation(im_color[:, :, (2, 1, 0)], label, return_rgb=True)
        # draw bounding boxes on it
        for i in range(bbox.shape[0]):
            x1 = int(bbox[i, 0])
            y1 = int(bbox[i, 1])
            x2 = int(bbox[i, 2])
            y2 = int(bbox[i, 3])
            # cv2.rectangle(im_label, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(im_label, "%.2f" % bbox[i, 4], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        rgb_msg = ros_numpy.msgify(Image, im_label, 'rgb8')
        rgb_msg.header.stamp = rgb_frame_stamp
        rgb_msg.header.frame_id = rgb_frame_id
        self.image_pub.publish(rgb_msg)

        if out_label_refined is not None:
            im_label_refined = visualize_segmentation(im_color[:, :, (2, 1, 0)], label_refined, return_rgb=True)
            rgb_msg_refined = ros_numpy.msgify(Image, im_label_refined, 'rgb8')
            rgb_msg_refined.header.stamp = rgb_frame_stamp
            rgb_msg_refined.header.frame_id = rgb_frame_id
            self.image_refined_pub.publish(rgb_msg_refined)
            
        # save results
        save_result = False
        if save_result:
            result = {'rgb': im_color, 'labels': label, 'labels_refined': label_refined}
            filename = os.path.join(self.output_dir, '%06d.mat' % self.counter)
            print(filename)
            scipy.io.savemat(filename, result, do_compression=True)
            filename = os.path.join(self.output_dir, '%06d.jpg' % self.counter)
            cv2.imwrite(filename, im_color)
            filename = os.path.join(self.output_dir, '%06d-label.jpg' % self.counter)
            cv2.imwrite(filename, im_label[:, :, (2, 1, 0)])
            filename = os.path.join(self.output_dir, '%06d-label-refined.jpg' % self.counter)
            cv2.imwrite(filename, im_label_refined[:, :, (2, 1, 0)])
            self.counter += 1
            sys.exit(1)


dirname = os.path.dirname(__file__)
def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a PoseCNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--pretrained', dest='pretrained',
                        help='initialize with pretrained checkpoint',
                        default=os.path.join(dirname, '../data/checkpoints/norm_model_0069999.pth'), type=str)
    parser.add_argument('--pretrained_crop', dest='pretrained_crop',
                        help='initialize with pretrained checkpoint for crops',
                        default=os.path.join(dirname, '../data/checkpoints/crop_dec9_model_final.pth'), type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', type=str)
    parser.add_argument('--network_cfg', dest='network_cfg_file',
                        help='config file for first stage network',
                        default=os.path.join(dirname, '../MSMFormer/configs/tabletop_pretrained.yaml'), type=str)
    parser.add_argument('--network_crop_cfg', dest='network_crop_cfg_file',
                        help='config file  for second stage network',
                        default=os.path.join(dirname, "../MSMFormer/configs/crop_tabletop_pretrained.yaml"), type=str)
    parser.add_argument('--dataset', dest='dataset_name',
                        help='dataset to train on',
                        default='shapenet_scene_train', type=str)
    parser.add_argument('--depth', dest='depth_name',
                        help='depth image pattern',
                        default='*depth.png', type=str)
    parser.add_argument('--color', dest='color_name',
                        help='color image pattern',
                        default='*color.png', type=str)
    parser.add_argument('--imgdir', dest='imgdir',
                        help='path of the directory with the test images',
                        default="data/demo", type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--network', dest='network_name',
                        help='name of the network',
                        default=None, type=str)
    parser.add_argument('--image_path', dest='image_path',
                        help='path to images', default=None, type=str)
    parser.add_argument('--input_image', dest='input_image',
                        help='the type of image', default="RGBD_ADD", type=str)
    parser.add_argument('--camera', dest='camera',
                        help='the type of image', default="Realsense", type=str)
    parser.add_argument('--no_refinement', dest='no_refinement',
                        help='do not use refinement',
                        action='store_true')                                                

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    print('Using config:')
    pprint.pprint(cfg)

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)

    # device
    cfg.gpu_id = 0
    cfg.device = torch.device('cuda:{:d}'.format(cfg.gpu_id))
    num_classes = 2
    cfg.MODE = 'TEST'
    cfg.TEST.VISUALIZE = False
    cfg.TEST.ROS_CAMERA = args.camera
    print('GPU device {:d}'.format(args.gpu_id))

    # prepare network
    predictor, cfg_transformer = get_predictor(cfg_file=args.network_cfg_file, weight_path=args.pretrained, input_image=args.input_image)
    if args.no_refinement:
        predictor_crop = None
        cfg_transformer_crop = None
    else:
        predictor_crop, cfg_transformer_crop = get_predictor_crop(cfg_file=args.network_crop_cfg_file, weight_path=args.pretrained_crop, input_image=args.input_image)

    # image listener
    listener = ImageListener(predictor, predictor_crop, cfg_transformer, cfg_transformer_crop)
    while not rospy.is_shutdown():
       listener.run_network()
