'''
    机器人控制代码, subscribe_gripper_angle_callback, subscribe_gripper_angle, close_gripper_num函数
    subscribe_gripper_angle_callback: 回调函数，返回夹爪状态
    subscribe_gripper_angle: 通过订阅话题执行回调函数
    close_gripper_num: 控制夹爪闭合程度(0~255) 0表示完全打开, 255表示完全闭合
'''

import sys
sys.path.append("/home/ywl/rekep_multicam/src/rekep_multicam/scripts")

import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
import sys
import tf
import numpy as np
from transform_utils import convert_pose_quat2mat, convert_pose_quat2euler, \
    convert_pose_mat2quat, convert_pose_quat2euler, convert_pose_euler2quat
from robotiq_2f_gripper_control.msg import _Robotiq2FGripper_robot_output as outputMsg
from robotiq_2f_gripper_control.msg import _Robotiq2FGripper_robot_input as inputMsg

import moveit_commander
import geometry_msgs.msg
import copy 
import torch
from ur_ikfast import ur_kinematics

# 机械臂操作相关代码
class RobotOperation():
    def __init__(self, Ttool2tcp):
        # rospy.init_node("UR10_Robot_Gripper_Publisher")
        self.trajectory_publihser = rospy.Publisher('/pos_joint_traj_controller/command', JointTrajectory, queue_size=10)
        self.gripper_publihser = rospy.Publisher('/Robotiq2FGripperRobotOutput', outputMsg.Robotiq2FGripper_robot_output, queue_size=10)
        self.UR10_joints = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",]
        # [X Y Z 三个欧拉角]，机械臂末端要运动到的位置
        self.goal_positions = []
        self.tf_listener = tf.TransformListener()
        # 夹爪是否抓取东西，是的话为1，不是的话为0
        self.gripper_state = 0.0
        self.Ttool2tcp = Ttool2tcp
        self.joint_angle = None
        rospy.sleep(1)
        self.init_gripper()
        self.get_joint_angle()      # 初始化之后就开始读取各个关节角
        self.ur10_arm = ur_kinematics.URKinematics('ur10')
        # 【新增】给 close_num 一个初始默认值 (0.0 表示打开)
        self.close_num = 0.0

    # [X Y Z 四元数]
    def get_UR10_pos(self, goal_positions):
        # 四元数转为欧拉角
        goal_positions = convert_pose_quat2euler(goal_positions[None])
        self.goal_positions = []
        for i in range(len(goal_positions)):
            self.goal_positions.append(float(goal_positions[i]))
    

    def UR10_moveto_angle(self, goal_angle):
        rospy.loginfo("Goal Position set lets go ! ")
        rospy.sleep(1)
        trajectory_msg = JointTrajectory()
        trajectory_msg.joint_names = self.UR10_joints
        trajectory_msg.points.append(JointTrajectoryPoint())
        trajectory_msg.points[0].positions = goal_angle
        trajectory_msg.points[0].velocities = [0.0 for i in self.UR10_joints]
        trajectory_msg.points[0].accelerations = [0.0 for i in self.UR10_joints]
        trajectory_msg.points[0].time_from_start = rospy.Duration(20)
        rospy.sleep(1)
        self.trajectory_publihser.publish(trajectory_msg)
        


    # 输入的是手抓位姿，但控制的是tool0的位置，不是手抓的
    def UR10_moveto_pose(self, target_positions:list, max_velocity_scale=0.1, TCP=True):
        moveit_commander.roscpp_initialize(sys.argv)
        move_group = moveit_commander.MoveGroupCommander("manipulator")

        # move_group.set_pose_reference_frame('base_link')
        move_group.set_max_acceleration_scaling_factor(0.001)
        move_group.set_max_velocity_scaling_factor(max_velocity_scale)
        end_effector_link = move_group.get_end_effector_link()      # tool0

        # 设置规划时间和允许误差,提升路径规划成功率
        move_group.set_planning_time(10.0)
        move_group.set_goal_tolerance(0.1)

        waypoints = []
        for target_position in target_positions:
            # base坐标系中的位姿，手抓的位姿
            base_target_pose = convert_pose_quat2mat(np.array(target_position))
            if TCP:
                # O为原点, A为tool0, B为TCP
                # T'OA TAB = TOA --> T'OA = TOA TBA
                base_target_pose = np.matmul(base_target_pose, self.Ttool2tcp)
            basetobaselink = np.array([[-1, 0, 0, 0],
                                       [0, -1, 0, 0],
                                       [0, 0, 1, 0],
                                       [0, 0, 0, 1]])
            base_link_target_pose = np.matmul(basetobaselink, base_target_pose)
            target_position = convert_pose_mat2quat(base_link_target_pose)

            target_pose = geometry_msgs.msg.Pose()
            target_pose.position.x = target_position[0]
            target_pose.position.y = target_position[1]
            target_pose.position.z = target_position[2]
            target_pose.orientation.x = target_position[3]
            target_pose.orientation.y = target_position[4]
            target_pose.orientation.z = target_position[5]
            target_pose.orientation.w = target_position[6]
            
            waypoints.append(copy.deepcopyconda(target_pose))
            
        current_pose = move_group.get_current_pose(end_effector_link).pose
        # 不要加起点，否则机械臂会出现卡顿的情况
        # waypoints.append(current_pose)
        # waypoints.append(copy.deepcopy(target_pose))


        # print(target_pose)
        # print(current_pose)

        fraction = 0.0   #路径规划覆盖率
        maxtries = 10   #最大尝试规划次数
        attempts = 0     #已经尝试规划次数
        eef_step = 0.01  # 路径分辨率（米）
        # # 设置机器臂当前的状态作为运动初始状态
        # move_group.set_start_state_to_current_state()

        # 尝试规划一条笛卡尔空间下的路径，依次通过所有路点
        while fraction < 1.0 and attempts < maxtries:
            (plan, fraction) = move_group.compute_cartesian_path(
                            waypoints,   # waypoint poses，路点列表
                            eef_step,        # eef_step，终端步进值
                            True)        # avoid_collisions，避障规划
            attempts += 1

            # current_pose = move_group.get_current_pose(end_effector_link).pose
            # current_pose_mat = convert_pose_quat2mat(np.array([current_pose.position.x, current_pose.position.y, current_pose.position.z,
            #                                 current_pose.orientation.x, current_pose.orientation.y, current_pose.orientation.z,
            #                                 current_pose.orientation.w]))
            # pose_matrix = self.get_ee_pose()
            # print("============================================")
            # print(pose_matrix)
            # print(np.matmul(current_pose_mat - pose_matrix))
            # exit()

            if attempts % 10 == 0:
                rospy.loginfo("Still trying after " + str(attempts) + " attempts...")
            
            if fraction >= 0.00:
                rospy.loginfo("Path computed successfully. Moving the arm.")
                move_group.execute(plan)
                rospy.loginfo("Path execution complete.")
                break
            
            else:
                rospy.loginfo("Path planning failed with only " + str(fraction) + " success after " + str(maxtries) + " attempts.")  
                rospy.sleep(1)
        rospy.sleep(1)


    def UR10_moveto_IKSolver(self, opt_pose_homo):
        moveit_commander.roscpp_initialize(sys.argv)
        arm = moveit_commander.MoveGroupCommander("manipulator")

        basetobaselink = np.array([[-1, 0, 0, 0],
                               [0, -1, 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]])
        base_link_opt_pose_homo = np.matmul(basetobaselink, opt_pose_homo)
        self.get_joint_angle()
        reset_joint_pos = self.joint_angle
        # IK Solver求解结果
        ik_solver_res = self.ur10_arm.inverse(convert_pose_mat2quat(base_link_opt_pose_homo), 
                                        False,
                                        q_guess = reset_joint_pos)
    
        arm.set_joint_value_target([q1, q2, q3, q4, q5, q6])
        plan = arm.plan()
        arm.execute(plan, wait=True)
        pass


    # TODO by DK
    # 设置机械臂各个关键的初始角
    def reset_joint_pos(self):
        init_angle = torch.Tensor([0.2200, -0.9412, -0.6413, 1.5519, 1.6567, -0.9322, 1.5342, 2.1447])
        pass


    # 获得末端执行器位姿
    # Twh hand到world的转换矩阵
    def get_ee_pose(self, return_quat=False):
        (trans, rot) = self.tf_listener.lookupTransform('/base', '/tool0_controller', rospy.Time(0))
        pose_numpy = np.array([trans[0], trans[1], trans[2], rot[0], rot[1], rot[2], rot[3]])
        pose_matrix = convert_pose_quat2mat(pose_numpy)
        if return_quat:
            return pose_numpy
        return pose_matrix  # [4 4]


    # 获得末端执行器三维坐标
    def get_ee_pos(self):
        (trans, rot) = self.tf_listener.lookupTransform('/base', '/tool0_controller', rospy.Time(0))
        pos_numpy = np.array([trans[0], trans[1], trans[2]])
        return pos_numpy  # [3]

    # 读取各个关节角的回调函数
    def get_joint_angle_callback(self, msg):
        gt_joint_name = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                         "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        joint_names = msg.name
        joint_positions = msg.position
        gt_joint_positions = []
        point_angle_dict = {}
        for name, joint_value in zip(joint_names, joint_positions):
            point_angle_dict[name] = joint_value
        for name in gt_joint_name:
            gt_joint_positions.append(point_angle_dict[name])
        self.joint_angle = np.array(gt_joint_positions)


    # numpy [6]
    def get_joint_angle(self):
        sub = rospy.Subscriber('/joint_states', JointState, self.get_joint_angle_callback)
        rospy.sleep(1)
    

    def open_gripper(self):
        print("Opening the gripper")
        rospy.sleep(1)
        gripper_value = outputMsg.Robotiq2FGripper_robot_output()
        gripper_value.rACT = 1
        gripper_value.rGTO = 1
        gripper_value.rATR = 0
        gripper_value.rPR = 0   # 爪子闭合程度, 0表示打开
        gripper_value.rSP = 255
        gripper_value.rFR = 150
        self.gripper_publihser.publish(gripper_value)
        rospy.sleep(1)
        self.gripper_state = 0.0


    def close_gripper(self):
        print("Closing the gripper")
        rospy.sleep(1)
        gripper_value = outputMsg.Robotiq2FGripper_robot_output()
        gripper_value.rACT = 1
        gripper_value.rGTO = 1
        gripper_value.rATR = 0
        gripper_value.rPR = 255 # 爪子闭合程度, 255表示闭合
        gripper_value.rSP = 255
        gripper_value.rFR = 150
        self.gripper_publihser.publish(gripper_value)
        rospy.sleep(1)
        self.gripper_state = 1.0

    # 爪子闭合一定的角度
    def close_gripper_num(self, clouse_num):
        rospy.sleep(1)
        gripper_value = outputMsg.Robotiq2FGripper_robot_output()
        gripper_value.rACT = 1
        gripper_value.rGTO = 1
        gripper_value.rATR = 0
        gripper_value.rPR = int(clouse_num) # 爪子闭合程度, 255表示闭合
        gripper_value.rSP = 255
        gripper_value.rFR = 150
        self.gripper_publihser.publish(gripper_value)
        rospy.sleep(1)
        self.gripper_state = 1.0


    # 获取夹爪开合角度的callback函数
    def subscribe_gripper_angle_callback(self, msg: inputMsg):
        MAX_COUNT = 255.0
        OPENING_PER_COUNT_MM = 0.4  # 2F-85: 0.4mm/count
        gpo = float(msg.gPO)  # 0(open) .. 255(closed)
        self.opening_mm = (MAX_COUNT - gpo) * OPENING_PER_COUNT_MM   # 距离
        self.opening_pct = (MAX_COUNT - gpo) / MAX_COUNT * 100.0     # 角度
        self.close_num = gpo

    def subscribr_gripper_angle(self):
        rospy.Subscriber("/Robotiq2FGripperRobotInput", inputMsg.Robotiq2FGripper_robot_input, 
                         self.subscribe_gripper_angle_callback, queue_size=10)
        rospy.sleep(1)


    def get_gripper_open_action(self):
        return 0.0
    

    def get_gripper_close_action(self):
        return 1.0

    def get_gripper_null_action(self):
        return self.gripper_state

    # 初始化robotiq机械爪
    def init_gripper(self):
        gripper_value = outputMsg.Robotiq2FGripper_robot_output()
        gripper_value.rACT = 0
        self.gripper_publihser.publish(gripper_value)


    def control_open_gripper(self):
        print("Opening the gripper")
        rospy.sleep(1)
        gripper_value = outputMsg.Robotiq2FGripper_robot_output()
        gripper_value.rACT = 1
        gripper_value.rGTO = 1
        gripper_value.rSP  = 255
        gripper_value.rFR  = 150
        self.gripper_publihser.publish(gripper_value)
        rospy.sleep(1)


    def control_close_gripper(self):
        print("Closing the gripper")
        rospy.sleep(1)
        gripper_value = outputMsg.Robotiq2FGripper_robot_output()
        gripper_value.rACT = 1
        gripper_value.rGTO = 1
        gripper_value.rATR = 0
        gripper_value.rPR = 255 # 爪子打开的角度, 255表示闭合
        gripper_value.rSP = 255
        gripper_value.rFR = 150
        self.gripper_publihser.publish(gripper_value)
        rospy.sleep(1)


    # TODO, 如何提取手抓和抓取物体的点云
    # (5000, 3)，碰撞点(手抓以及抓取物体的点云)坐标 Get the points of the gripper and any object in hand.
    # 手抓坐标系中的坐标
    def get_collision_points(self):
        return np.array([[0.0, 0.0, 0.0]])

    # TODO, 如何生成体素
    def get_sdf_voxels(self, sdf_voxel_size):
        return None


    # TODO: asscociate keypoints with closest object (mask?)
    # 根据关键点获得关键点所属的物体
    def get_object_by_keypoint(self, index):
        return None

    # TODO: How to judge which keypoints are grasped?
    def is_grasping(self, candidate_obj=None):
        """Check if gripper is grasping"""
        # Could be enhanced with force sensor readings
        # TODO, how to modify this
        print("Yes it is grasping")
        return self.gripper_state == 1.0

    def create_approach_pose(self, grasp_pose_quat, height_offset=0.3):
        """创建接近位姿（在抓取点上方指定高度）"""
        grasp_pose_mat = convert_pose_quat2mat(grasp_pose_quat)
        approach_pose_mat = grasp_pose_mat.copy()
        approach_pose_mat[2, 3] += height_offset
        return convert_pose_mat2quat(approach_pose_mat)


if __name__ == "__main__":
    rospy.init_node("UR10_Robot_Gripper_Publisher")
    tf_listener = tf.TransformListener()
    rospy.sleep(1)
    (trans, rot) = tf_listener.lookupTransform('/tool0_controller', '/tool0', rospy.Time(0))
    Ttool2tcp = np.array([trans[0], trans[1], trans[2], rot[0], rot[1], rot[2], rot[3]])
    Ttool2tcp = convert_pose_quat2mat(Ttool2tcp)
    robot_move = RobotOperation(Ttool2tcp)

    robot_move.subscribr_gripper_angle()
    print(robot_move.opening_mm, robot_move.opening_pct, robot_move.close_num)

    robot_move.get_joint_angle()
    print("joint angle:", robot_move.joint_angle)

    ee_pose = robot_move.get_ee_pose(return_quat = True)
    print("End Eff Pose:", ee_pose)

    for i in range(2):
        robot_move.close_gripper_num(i * 40)

    init_sol = robot_move.get_ee_pose(return_quat = True)
    print(f"初始位姿:{init_sol}")
    # robot_move.UR10_moveto_pose([[-0.10335114, 0.6072646, 0.64588992, -0.94640856, -0.06480774, -0.0466047, 0.31295176]])
    #robot_move.UR10_moveto_pose(list(np.array([[0.01518182,  0.00179752, -0.09793237,  0.16321577, -0.02193037,0.12723099,  0.97810631]])))
##########新的
    #robot_move.UR10_moveto_pose([[0.00879758,  0.01828052, -0.1097819 ,  0.13026495,  0.16782452,-0.46541848,  0.85921569]])
##########旧的

    # robot_move.UR10_moveto_pose([[2.40322528e-01, 7.59688154e-01, 2.49989794e-01, -9.95904157e-01, 2.57143947e-02, 1.30725943e-04, 8.66813917e-02]])
    robot_move.UR10_moveto_pose([[-0.01342209,  0.71531992,  0.34592575,  0.89852276,  0.38075121, -0.08759944, -0.20002926]])