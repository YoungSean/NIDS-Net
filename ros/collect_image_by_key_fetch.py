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
import ros_numpy

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
        self.ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size, slop_seconds)
        rospy.sleep(3.0)
        input('start capturing?')
        # ts.registerCallback(self.callback)
        self.ts.registerCallback(self.sync_callback)

        # To store the latest synchronized images
        self.latest_rgb = None
        self.latest_depth = None

    def sync_callback(self, rgb, depth):
        # Store the latest synchronized images
        self.latest_rgb = rgb
        self.latest_depth = depth
        print("Synchronized images received.")

    def save_images(self):
        if self.latest_rgb is None or self.latest_depth is None:
            print("No synchronized images to save.")
            return
        
        # Save RGB image
        # rgb_image = self.cv_bridge.imgmsg_to_cv2(self.latest_rgb, "bgr8")
        rgb_image = ros_numpy.numpify(self.latest_rgb)
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        rgb_filename = osp.join(self.save_dir, f'color-{self.count:06d}.jpg')
        cv2.imwrite(rgb_filename, rgb_image)
        
        # Save Depth image
        if self.latest_depth.encoding == '32FC1':
            depth_cv = ros_numpy.numpify(self.latest_depth)
        elif self.latest_depth.encoding == '16UC1':
            depth_cv = ros_numpy.numpify(self.latest_depth).copy().astype(np.float32)
            depth_cv /= 1000.0
        else:
            rospy.logerr_throttle(
                1, 'Unsupported depth type. Expected 16UC1 or 32FC1, got {}'.format(
                    self.latest_depth.encoding))
            return
        depth_filename = osp.join(self.save_dir, f'depth-{self.count:06d}.png')
        cv2.imwrite(depth_filename, depth_cv)

        print(f"Saved RGB image to {rgb_filename}")
        print(f"Saved Depth image to {depth_filename}")
        self.count += 1
        self.latest_rgb = None
        self.latest_depth = None

if __name__ == '__main__':
    listener = ImageListener()
    try:
        while not rospy.is_shutdown():
            input("Press Enter to save the latest synchronized images...")
            listener.save_images()
    except KeyboardInterrupt:
        print("Shutting down")