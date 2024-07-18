#!/usr/bin/env python

# --------------------------------------------------------
# PoseCNN
# Copyright (c) 2018 NVIDIA
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""collect images from Fetch"""

import rospy
import message_filters
import cv2
import argparse
import pprint
import time, os, sys
import os.path as osp
import numpy as np
import datetime
import tf2_ros
import scipy.io
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo
from tf.transformations import quaternion_from_euler, quaternion_matrix

class ImageListener:

    def __init__(self):

        self.cv_bridge = CvBridge()
        self.count = 0
        self.base_frame = 'base_link'
        self.camera_frame = 'head_camera_rgb_optical_frame'

        # output dir
        this_dir = osp.dirname(__file__)
        self.outdir = osp.join(this_dir, 'data_fetch')
        if not osp.exists(self.outdir):
            os.mkdir(self.outdir)

        now = datetime.datetime.now()
        seq_name = "{:%m%dT%H%M%S}/".format(now)
        self.save_dir = osp.join(self.outdir, seq_name)
        if not osp.exists(self.save_dir):
            os.mkdir(self.save_dir)

        # initialize a node
        rospy.init_node("image_listener")
        rgb_sub = message_filters.Subscriber('/head_camera/rgb/image_raw', Image, queue_size=2)
        depth_sub = message_filters.Subscriber('/head_camera/depth_registered/image_raw', Image, queue_size=2)
        
        # camera parameters
        msg = rospy.wait_for_message('/head_camera/rgb/camera_info', CameraInfo)
        # update camera intrinsics
        intrinsics = np.array(msg.K).reshape(3, 3)
        self.intrinsics = intrinsics
        print(intrinsics)
        
        # camera pose in base
        tf_buffer = tf2_ros.Buffer(rospy.Duration(100.0))  # tf buffer length
        tf_listener = tf2_ros.TransformListener(tf_buffer)
        rospy.sleep(1.0)
        transform = tf_buffer.lookup_transform(self.base_frame,
                                           # source frame:
                                           self.camera_frame,
                                           # get the tf at the time the pose was valid
                                           rospy.Time.now(),
                                           # wait for at most 1 second for transform, otherwise throw
                                           rospy.Duration(1.0)).transform
        quat = [transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w]
        RT = quaternion_matrix(quat)
        RT[0, 3] = transform.translation.x
        RT[1, 3] = transform.translation.y        
        RT[2, 3] = transform.translation.z
        self.camera_pose = RT
        print(self.camera_pose)
        
        # save meta data
        factor_depth = 1000.0
        meta = {'intrinsic_matrix': self.intrinsics, 'factor_depth': factor_depth, 'camera_pose': self.camera_pose}
        filename = self.save_dir + 'meta-data.mat'
        scipy.io.savemat(filename, meta, do_compression=True)
        print('save data to {}'.format(filename))        

        queue_size = 1
        slop_seconds = 0.025
        ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size, slop_seconds)
        rospy.sleep(3.0)
        input('start capturing?')
        ts.registerCallback(self.callback)

    def callback(self, rgb, depth):
        if depth.encoding == '32FC1':
            depth_32 = self.cv_bridge.imgmsg_to_cv2(depth) * 1000
            depth_cv = np.array(depth_32, dtype=np.uint16)
        elif depth.encoding == '16UC1':
            depth_cv = self.cv_bridge.imgmsg_to_cv2(depth)
        else:
            rospy.logerr_throttle(
                1, 'Unsupported depth type. Expected 16UC1 or 32FC1, got {}'.format(
                    depth.encoding))
            return

        # write images
        im = self.cv_bridge.imgmsg_to_cv2(rgb, 'bgr8')
        filename_color = self.save_dir + 'color-%06d.jpg' % self.count
        filename_depth = self.save_dir + 'depth-%06d.png' % self.count
        if self.count % 1 == 0:
            cv2.imwrite(filename_color, im)
            cv2.imwrite(filename_depth, depth_cv)
            print(filename_color)
            print(filename_depth)
        self.count += 1


if __name__ == '__main__':

    # image listener
    listener = ImageListener()
    try:  
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
