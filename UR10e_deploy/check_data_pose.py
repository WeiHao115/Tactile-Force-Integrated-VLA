from robot_control import RobotOperation
import rospy
import tf
import numpy as np
from transform_utils import convert_pose_quat2mat, convert_pose_quat2euler, \
    convert_pose_mat2quat, convert_pose_quat2euler, convert_pose_euler2quat
import time


rospy.init_node("UR10_Robot_Gripper_Publisher")
tf_listener = tf.TransformListener()
rospy.sleep(1)
(trans, rot) = tf_listener.lookupTransform('/tool0_controller', '/tool0', rospy.Time(0))
Ttool2tcp = np.array([trans[0], trans[1], trans[2], rot[0], rot[1], rot[2], rot[3]])
Ttool2tcp = convert_pose_quat2mat(Ttool2tcp)
robotoperation = RobotOperation(Ttool2tcp)

type = "angle"
if type == "ee":
    all_pose = np.loadtxt("/home/k202/0604_dk/000000/ur10_ee_pose.txt")[:, 1:]
    start_time = time.time()
    for index, single_pose in enumerate(all_pose):
        if index == 614:
            print(time.time() - start_time, "s")
            exit()
        robotoperation.UR10_moveto_pose_rtde([single_pose])
if type == "angle":
    all_pose = np.loadtxt("/home/k202/0604_dk/000000/ur10_angle.txt")[:, 1:]
    start_time = time.time()
    for index, single_pose in enumerate(all_pose):
        if index == 614:
            print(time.time() - start_time, "s")
            exit()
        robotoperation.UR10_moveto_angle_rtde(single_pose)














