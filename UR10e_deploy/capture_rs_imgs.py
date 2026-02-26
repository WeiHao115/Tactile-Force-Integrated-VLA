'''
    用于读取realsense采集的RGB图像和深度图
    由于cv_bridge不支持python3.10, 转用了ros_numpy
'''

import sys
import rospy
import message_filters
from sensor_msgs.msg import Image
import cv2
import matplotlib.pylab as plt
import numpy as np
import os

if not hasattr(np, "float"):
    np.float = float
import ros_numpy


class CameraFeedManager:
    def __init__(self):
        self.camera_K = np.array([
            382.85443115234375, 0.0, 323.7813720703125,
            0.0, 381.9345703125, 238.988525390625,
            0.0, 0.0, 1.0
        ]).reshape(3, 3)

    @staticmethod
    def imgmsg_to_bgr(msg):
        img = ros_numpy.numpify(msg)  # HxW(xC)

        enc = (msg.encoding or "").lower()
        if enc == "rgb8":
            # RGB -> BGR
            img = img[..., ::-1]
        elif enc == "rgba8":
            # RGBA -> BGR
            img = img[..., :3][..., ::-1]
        elif enc == "bgra8":
            # BGRA -> BGR
            img = img[..., :3]
        return np.ascontiguousarray(img)

    @staticmethod
    def imgmsg_to_depth(msg):
        depth = ros_numpy.numpify(msg)
        return np.ascontiguousarray(depth)

    def get_latest_frames(self):
        rgb_msg = rospy.wait_for_message("/camera/color/image_raw", Image)
        # dep_msg = rospy.wait_for_message("/camera/aligned_depth_to_color/image_raw", Image)
        rgb = self.imgmsg_to_bgr(rgb_msg)
        # dep = self.imgmsg_to_depth(dep_msg)
        return rgb, rgb, rgb, rgb

    # ########### Modified by Weihao
    # def get_latest_frames(self):
    #     # 1. 获取外部相机数据 
    #     rgb_msg_ext = rospy.wait_for_message("/third_pov/color/image_raw", Image)
    #     dep_msg_ext = rospy.wait_for_message("/third_pov/aligned_depth_to_color/image_raw", Image)
        
    #     # 2. 获取腕部相机数据 
    #     rgb_msg_wrist = rospy.wait_for_message("/wrist_cam/color/image_raw", Image)
    #     dep_msg_wrist = rospy.wait_for_message("/wrist_cam/aligned_depth_to_color/image_raw", Image)

    #     # 3. 转换格式
    #     rgb_ext = self.imgmsg_to_bgr(rgb_msg_ext)
    #     dep_ext = self.imgmsg_to_depth(dep_msg_ext)
    #     rgb_wrist = self.imgmsg_to_bgr(rgb_msg_wrist)
    #     dep_wrist = self.imgmsg_to_depth(dep_msg_wrist)

    #     return rgb_ext, dep_ext, rgb_wrist, dep_wrist

    def calcul_kpts_w_coords(self, rgb_img, dep_img, K, Twc):
        depth_mask = dep_img != 0
        H, W, C = rgb_img.shape
        grid_x, grid_y  = np.meshgrid(np.arange(W), np.arange(H)) 
        effec_rgb_pixels = np.concatenate((grid_x[..., None], grid_y[..., None]), axis=-1)[depth_mask]
        effec_rgb_pixels = np.concatenate((effec_rgb_pixels, np.ones((effec_rgb_pixels.shape[0], 1))), axis=-1)
        effec_depth = dep_img[depth_mask][:, None] / 1000.0  # [N 1]
        P_c = np.matmul(np.linalg.inv(K), (effec_rgb_pixels * effec_depth).transpose(1, 0))
        P_w = (np.matmul(Twc[:3, :3], P_c) + Twc[:3, [-1]]).transpose(1, 0)     # [N 3]
        return P_w

# if __name__ == "__main__":
#     rospy.init_node("realsense_node", anonymous=True)
#     camerafeedmanager = CameraFeedManager()
#     rgb_img, dep_img = camerafeedmanager.get_latest_frames()
#     cv2.imwrite("./rgb_img.png", rgb_img)
#     cv2.imwrite("./dep_img.png", dep_img)
#     camera_K = np.array([382.85443115234375, 0.0, 323.7813720703125, 0.0, 381.9345703125, 238.988525390625, 0.0, 0.0, 1.0]).reshape(3, 3)
#     camera_Twc = np.array([[1, 0, 0, 0],
#                            [0, 1, 0, 0],
#                            [0, 0, 1, 0],
#                            [0, 0, 0, 1]])
#     P_w = camerafeedmanager.calcul_kpts_w_coords(ext_rgb, dep_img, camera_K, camera_Twc)
#     np.savetxt("./scene.txt", P_w)



if __name__ == "__main__":
    rospy.init_node("realsense_node", anonymous=True)
    camerafeedmanager = CameraFeedManager()
    
    # 接收四个返回值
    ext_rgb, ext_dep, wrist_rgb, wrist_dep = camerafeedmanager.get_latest_frames()
    
    # 保存图片测试
    cv2.imwrite("./ext_rgb.png", ext_rgb)
    cv2.imwrite("./wrist_rgb.png", wrist_rgb)
    
    camera_K = np.array([382.85443115234375, 0.0, 323.7813720703125, 0.0, 381.9345703125, 238.988525390625, 0.0, 0.0, 1.0]).reshape(3, 3)
    camera_Twc = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
    P_w = camerafeedmanager.calcul_kpts_w_coords(ext_rgb, ext_dep, camera_K, camera_Twc)
    np.savetxt("./scene.txt", P_w)
    print("Scene point cloud saved using External Camera data.")



