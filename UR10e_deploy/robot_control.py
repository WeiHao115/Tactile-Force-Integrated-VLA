import sys
sys.path.append("/home/ywl/rekep_multicam/src/rekep_multicam/scripts")
import ast
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

import math


# 机械臂操作相关代码
class RobotOperation():
    def __init__(self, Ttool2tcp):
        # rospy.init_node("UR10_Robot_Gripper_Publisher")
        self.trajectory_publihser = rospy.Publisher('/scaled_pos_joint_traj_controller/command', JointTrajectory, queue_size=10)
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
        self.init_move_class()
    

    # [X Y Z 四元数]
    def get_UR10_pos(self, goal_positions):
        # 四元数转为欧拉角
        goal_positions = convert_pose_quat2euler(goal_positions[None])
        self.goal_positions = []
        for i in range(len(goal_positions)):
            self.goal_positions.append(float(goal_positions[i]))
    

    def init_move_class(self, max_velocity_scale = 0.1):
        moveit_commander.roscpp_initialize(sys.argv)
        self.move_group = moveit_commander.MoveGroupCommander("manipulator")

        # move_group.set_pose_reference_frame('base_link')
        self.move_group.set_max_acceleration_scaling_factor(0.001)
        self.move_group.set_max_velocity_scaling_factor(max_velocity_scale)
        self.end_effector_link = self.move_group.get_end_effector_link()      # tool0

        # 设置规划时间和允许误差,提升路径规划成功率
        self.move_group.set_planning_time(10.0)
        self.move_group.set_goal_tolerance(0.1)


    # 输入的是手抓位姿，但控制的是tool0的位置，不是手抓的
    def UR10_moveto_pose(self, target_positions:list, max_velocity_scale=0.1, TCP=True):
        # moveit_commander.roscpp_initialize(sys.argv)
        # move_group = moveit_commander.MoveGroupCommander("manipulator")

        # # move_group.set_pose_reference_frame('base_link')
        # move_group.set_max_acceleration_scaling_factor(0.001)
        # move_group.set_max_velocity_scaling_factor(max_velocity_scale)
        # end_effector_link = move_group.get_end_effector_link()      # tool0

        # # 设置规划时间和允许误差,提升路径规划成功率
        # move_group.set_planning_time(10.0)
        # move_group.set_goal_tolerance(0.1)

        waypoints = []
        for target_position in target_positions:
            # base坐标系中的位姿，手抓的位姿
            base_target_pose = convert_pose_quat2mat(np.array(target_position))
            if TCP:
                # O为原点, A为tool0, B为TCP
                # T'OA TAB = TOA  --> T'OA = TOA TBA
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
            
            waypoints.append(copy.deepcopy(target_pose))
            
        current_pose = self.move_group.get_current_pose(self.end_effector_link).pose
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
            (plan, fraction) = self.move_group.compute_cartesian_path(
                            waypoints,   # waypoint poses，路点列表
                            eef_step,        # eef_step，终端步进值
                            True)        # avoid_collisions，避障规划
            attempts += 1
            if attempts % 10 == 0:
                rospy.loginfo("Still trying after " + str(attempts) + " attempts...")
            
            if fraction >= 0.00:
                rospy.loginfo("Path computed successfully. Moving the arm.")
                self.move_group.execute(plan)
                rospy.loginfo("Path execution complete.")
                break
            
            else:
                rospy.loginfo("Path planning failed with only " + str(fraction) + " success after " + str(maxtries) + " attempts.")  
                rospy.sleep(1)
        # rospy.sleep(1)



    def UR10_moveto_angle(self, goal_angle):
        rospy.loginfo("Goal Position set lets go ! ")
        # rospy.sleep(0.1)
        trajectory_msg = JointTrajectory()
        trajectory_msg.joint_names = self.UR10_joints
        trajectory_msg.points.append(JointTrajectoryPoint())
        trajectory_msg.points[0].positions = goal_angle
        trajectory_msg.points[0].velocities = [0.0 for i in self.UR10_joints]
        trajectory_msg.points[0].accelerations = [0.0 for i in self.UR10_joints]
        trajectory_msg.points[0].time_from_start = rospy.Duration(1)
        # rospy.sleep(0.1)
        self.trajectory_publihser.publish(trajectory_msg)


    def UR10_moveto_IKSolver(self, opt_pose_quat):
        moveit_commander.roscpp_initialize(sys.argv)
        arm = moveit_commander.MoveGroupCommander("manipulator")

        basetobaselink = np.array([[-1, 0, 0, 0],
                                   [0, -1, 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]])
        base_link_opt_pose_homo = np.matmul(basetobaselink, convert_pose_quat2mat(np.array(opt_pose_quat)))
        self.get_joint_angle()
        reset_joint_pos = self.joint_angle
        # IK Solver求解结果
        ik_solver_res = self.ur10_arm.inverse(convert_pose_mat2quat(base_link_opt_pose_homo), 
                                        False,
                                        q_guess = reset_joint_pos)
        import pdb; pdb.set_trace()
        self.UR10_moveto_angle(ik_solver_res)

    

    
    #-------------------------------------------------------------------------------------------
    # TODO by DK -> MODIFIED by Gemini
    # 设置机械臂各个关键的初始角
    def reset_joint_pos(self, duration_sec=5.0):
        """
        将机械臂移动到一个预定义的、安全的 "Home" 位置。
        使用 UR10_moveto_angle 方法执行。
        """
        rospy.loginfo("正在执行复位: 移动到 'Home' 姿态...")

        # 一个常见的、安全的 UR10 "Home" 姿态 (所有关节弯曲，指向前方)
        # 您可以根据需要修改这些值
        home_angle = [
            0.0,                      # shoulder_pan_joint
            -math.pi / 2.0,            # shoulder_lift_joint
            math.pi / 2.0,            # elbow_joint
            -math.pi / 2.0,            # wrist_1_joint
            -math.pi / 2.0,            # wrist_2_joint
            0.0                       # wrist_3_joint
        ]

        # --- 复用您的 UR10_moveto_angle 函数逻辑 ---
        # (基于 UR10_moveto_angle 函数)
        rospy.loginfo("目标 'Home' 角度: %s", [round(a, 2) for a in home_angle])

        trajectory_msg = JointTrajectory()
        trajectory_msg.joint_names = self.UR10_joints
        trajectory_msg.points.append(JointTrajectoryPoint())
        trajectory_msg.points[0].positions = home_angle
        trajectory_msg.points[0].velocities = [0.0 for i in self.UR10_joints]
        trajectory_msg.points[0].accelerations = [0.0 for i in self.UR10_joints]

        # 使用一个合理的运动时间，例如 5 秒
        # 您在 UR10_moveto_angle 中硬编码了 20 秒，这里我们用一个参数
        trajectory_msg.points[0].time_from_start = rospy.Duration(duration_sec)

        rospy.sleep(1) #
        self.trajectory_publihser.publish(trajectory_msg)
        rospy.loginfo("'Home' 姿态指令已发送。")

        # 注意: 原生的 UR10_moveto_angle 没有等待执行完毕的逻辑。
        # 为简单起见，这里也直接发送指令。
        # 我们 sleep 一下，等待运动开始。
        rospy.sleep(duration_sec + 0.5)
        rospy.loginfo("复位动作应已完成。")


    #--------------------------------------------------------------------------------------------
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
        rospy.sleep(0.1) 
        gripper_value = outputMsg.Robotiq2FGripper_robot_output()
        gripper_value.rACT = 1
        gripper_value.rGTO = 1
        gripper_value.rATR = 0
        gripper_value.rPR = 0   # 爪子打开的角度, 0表示打开
        gripper_value.rSP = 255
        gripper_value.rFR = 150
        self.gripper_publihser.publish(gripper_value)
        rospy.sleep(0.1)
        self.gripper_state = 0.0


    def close_gripper(self):
        print("Closing the gripper")
        rospy.sleep(0.1)
        gripper_value = outputMsg.Robotiq2FGripper_robot_output()
        gripper_value.rACT = 1
        gripper_value.rGTO = 1
        gripper_value.rATR = 0
        gripper_value.rPR = 255 # 爪子打开的角度, 255表示闭合
        gripper_value.rSP = 255
        gripper_value.rFR = 150
        self.gripper_publihser.publish(gripper_value)
        rospy.sleep(0.1)
        self.gripper_state = 1.0


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
        rospy.sleep(0.2)

    # 爪子闭合一定的角度
    def close_gripper_num(self, clouse_num):
        rospy.sleep(0.2)
        gripper_value = outputMsg.Robotiq2FGripper_robot_output()
        gripper_value.rACT = 1
        gripper_value.rGTO = 1
        gripper_value.rATR = 0
        gripper_value.rPR = int(clouse_num) # 爪子闭合程度, 255表示闭合
        gripper_value.rSP = 255
        gripper_value.rFR = 150
        self.gripper_publihser.publish(gripper_value)
        rospy.sleep(0.2)
        self.gripper_state = 1.0


    # def load_waypoints_from_txt(self,file_path):
    #     try:
    #         raw_data = np.loadtxt(file_path, dtype=float)
    #         poses_data = raw_data[:, :]
    #         if poses_data.shape[1] != 7:
    #             print(f"[警告] 数据维度不对! 期望 7 列数据, 实际读取到 {poses_data.shape[1]} 列")
    #             return []
    #         return poses_data.tolist()
    #     except Exception as e:
    #         print(f"[错误] 读取文件失败: {e}")
    #         return []


if __name__ == "__main__":
    rospy.init_node("UR10_Robot_Gripper_Publisher")
    tf_listener = tf.TransformListener()
    rospy.sleep(1)
    (trans, rot) = tf_listener.lookupTransform('/tool0_controller', '/tool0', rospy.Time(0))
    Ttool2tcp = np.array([trans[0], trans[1], trans[2], rot[0], rot[1], rot[2], rot[3]])
    Ttool2tcp = convert_pose_quat2mat(Ttool2tcp)
    
    robotoperation = RobotOperation(Ttool2tcp)
    pose = robotoperation.get_ee_pose(return_quat = True)
    print(pose)# reset_reg_cost         : 1.30896

    robotoperation.open_gripper()
    #robotoperation.close_gripper_num(0)
    #robotoperation.close_gripper_num(229)
    robotoperation.close_gripper()
   
    # robot_move = RobotOperation(Ttool2tcp)
    # robot_move.get_joint_angle()
    # print("joint angle:", robot_move.joint_angle)

    # ee_pose = robot_move.get_ee_pose(return_quat = True)
    # print("End Eff Pose:", ee_pose)
    # # exit()

    # init_sol = robot_move.get_ee_pose(return_quat = True)
    
    # init_sol = convert_pose_quat2euler(init_sol)
    # print(f"优化初值:{init_sol}")
    ###################初始位姿
    # robotoperation.UR10_moveto_pose([[-0.31895895, 0.66285471, 0.51663578, -0.93785405, -0.17891105, -0.02265816, 0.29649151]])

    # robotoperation.UR10_moveto_pose([[-0.38784, 1.10571, 0.155, -0.93785405, -0.17891105, -0.02265816, 0.29649151]])
    # 0.3878443289161494 1.1057178609571824 0.0562189455362975 0.0005169622706755389 0.0018612333046035615 0.03303484005805276 0.9994523339824344
    # 0.010583956188920354 1.1948588423261886 0.5693286405499571 -0.24228574951164117 -0.0046518278200046795 0.618717115664556 0.7473052300534553

    # all_pose = np.loadtxt("/home/ywl/test_arr_tcp_abs.txt")[::10]
    # for i in range(all_pose.shape[0]):
    #     robotoperation.UR10_moveto_pose([all_pose[i]])
    # robotoperation.UR10_moveto_pose([[-0.31895895, 0.66285471, 0.51663578, -0.93785405, -0.17891105, -0.02265816, 0.29649151]])


    # all_pose = np.loadtxt("/home/ywl/test_arr_tcp_abs.txt")
    # robotoperation.UR10_moveto_pose([list(all_pose[i]) for i in range(all_pose.shape[0])])

    robotoperation.UR10_moveto_pose([[-0.31895895, 0.66285471, 0.51663578, -0.93785405, -0.17891105, -0.02265816, 0.29649151]])

    # robotoperation.UR10_moveto_pose([[0.4164185223500147,0.9225218656363163 ,0.16892100383971034 ,0.2556119373946606, 0.02008919164768721, 0.17438631902847515 ,0.9507094054315385]])
    # robotoperation.UR10_moveto_pose([[0.41641704263647306 ,0.9225078178266881 ,0.16892395072567853, 0.25559897861529257, 0.020103519209294346 ,0.1743831009763194, 0.9507131769046350]])
    # robotoperation.UR10_moveto_pose([[0.4164056275116815, 0.9223152954045618, 0.16889177531676525, 0.25566456114002484, 0.020068984418372654 ,0.174345697769888, 0.950703132271386]])
    # robotoperation.UR10_moveto_pose([[0.4261588168587595, 0.858629618780049 ,0.28407404992566365, 0.27536550958045225, 0.030122431313555357 ,0.10480240849004223 ,0.9551350325686647]])
    # robotoperation.UR10_moveto_pose([[0.4296073008943968, 0.8558510323506989 ,0.4649540877949935, 0.24707785638271693 ,0.003535023695799816 ,0.05535622196628423, 0.9674067010220733]])
    # robotoperation.UR10_moveto_pose([[0.471669775079788, 0.8634773692973863 ,0.6307909312900398 ,0.1784911133960218 ,-0.035334020008058596 ,-0.0013189905696931999 ,0.9833060000491178]])
    # robotoperation.UR10_moveto_pose([[0.10820728248278966, 1.0221518527674616 ,0.6770045011381005, -0.23271406234606176 ,0.0026586631833536547 ,0.4910404059768488 ,0.8394738926223747]])
    # robotoperation.UR10_moveto_pose([[-0.013143427730650786, 1.1353800865050607, 0.2908637865038792 ,-0.30946329927261756 ,-0.14151736132493462 ,0.24029855180230503, 0.9090995043715783]])
    # robotoperation.UR10_moveto_pose([[0.10174775419753514 ,1.0755037290199299, 0.6471738274224285 ,-0.23618077592226983 ,0.0887110834746816 ,0.46710271079830507, 0.8474455984417983]])
    # robotoperation.UR10_moveto_pose([[0.35234139004689013 ,0.8778721676415399, 0.25304975822137216 ,0.015288511255998442 ,-0.10547993230518507 ,0.17648409459804176 ,0.9785160242215649]])
    # robotoperation.UR10_moveto_pose([[0.3904330366915849 ,0.9652781041800368, 0.19531659649489483 ,-0.007276159546751315 ,0.003069760118735699 ,0.08711347102951518 ,0.996167093032215]])