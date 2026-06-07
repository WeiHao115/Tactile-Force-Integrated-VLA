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

import math

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

import moveit_commander
import geometry_msgs.msg
import copy 
from std_msgs.msg import Float32, Int32
import math




# 机械臂操作相关代码
class RobotOperation():
    def __init__(self, Ttool2tcp):

        self.MAX_REGISTER = 1000.0
        self.MAX_STROKE_MM = 200.0
        self.current_pos_register = -1
        self.opening_mm = self.MAX_STROKE_MM
        self.opening_pct = 0.0
        self.close_num = 0.0
        self.gripper_state = 0.0

        # rospy.init_node("UR10_Robot_Gripper_Publisher")
        self.trajectory_publihser = rospy.Publisher('/scaled_pos_joint_traj_controller/command', JointTrajectory, queue_size=10)
        self.gripper_publihser = rospy.Publisher('/Robotiq2FGripperRobotOutput', outputMsg.Robotiq2FGripper_robot_output, queue_size=10)
        #rospy.init_node('dh_gripper_python_client', anonymous=True)
        self.pub_force = rospy.Publisher('/gripper/close_with_force', Float32, queue_size=10)
        self.pub_pos_mm = rospy.Publisher('/gripper/set_pos_mm', Float32, queue_size=10)
        self.sub_status = rospy.Subscriber('/gripper/curr_pos', Int32, self.subscribr_gripper_angle)

        rospy.sleep(1.0)
        rospy.loginfo("DH Gripper Client Initialized.")
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
        # self.init_gripper()
        self.get_joint_angle()      # 初始化之后就开始读取各个关节角
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

    # 读取各个关节角的回调函数rospy
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
    

    # def init_gripper(self):
    #     rospy.init_node('dh_gripper_python_client', anonymous=True)
    #     self.pub_force = rospy.Publisher('/gripper/close_with_force', Float32, queue_size=10)
    #     self.pub_pos_mm = rospy.Publisher('/gripper/set_pos_mm', Float32, queue_size=10)
    #     self.sub_status = rospy.Subscriber('/gripper/curr_pos', Int32, self.subscribr_gripper_angle)

    #     rospy.sleep(1.0)
    #     rospy.loginfo("DH Gripper Client Initialized.")

    def subscribr_gripper_angle(self, msg):
        #大寰的底层物理逻辑是0 mm 代表完全闭合，80 mm 代表完全张开，但是我们的定义是0 代表完全张开，80 代表完全闭合80mm，所以要进行转换
        current_register = float(msg.data)
        self.current_pos_register = current_register
        self.opening_mm = (current_register / self.MAX_REGISTER) * self.MAX_STROKE_MM
        self.opening_pct = 100.0 - ((current_register / self.MAX_REGISTER) * 100.0)
        self.close_num = self.opening_pct
        
    def close_with_force(self, target_force_n):
        rospy.loginfo("Close with force %.1f N", target_force_n)
        msg = Float32()
        msg.data = float(target_force_n)
        self.pub_force.publish(msg)

    def close_with_pos(self, target_pos_mm):
        rospy.loginfo("Close with position %.1f mm", target_pos_mm)
        # if target_pos_mm < 10:
        #     target_pos_mm = 10
        msg = Float32()
        msg.data = float(target_pos_mm)

        self.pub_pos_mm.publish(msg)
        rospy.sleep(0.1)

    def close_gripper_num(self, clouse_num):
        clouse_num = max(0.0, min(float(clouse_num), 100.0))
        target_mm = self.MAX_STROKE_MM * (1.0 - (clouse_num / 100.0))
        msg = Float32()
        msg.data = target_mm
        self.pub_pos_mm.publish(msg)
        
        rospy.sleep(3)
        self.gripper_state = 1.0


    # def open_gripper(self):
    #     print("Opening the gripper")
    #     rospy.sleep(0.1) 
    #     gripper_value = outputMsg.Robotiq2FGripper_robot_output()
    #     gripper_value.rACT = 1
    #     gripper_value.rGTO = 1
    #     gripper_value.rATR = 0
    #     gripper_value.rPR = 0   # 爪子打开的角度, 0表示打开
    #     gripper_value.rSP = 255
    #     gripper_value.rFR = 150
    #     self.gripper_publihser.publish(gripper_value)
    #     rospy.sleep(0.1)
    #     self.gripper_state = 0.0


    # def close_gripper(self):
    #     print("Closing the gripper")
    #     rospy.sleep(0.1)
    #     gripper_value = outputMsg.Robotiq2FGripper_robot_output()
    #     gripper_value.rACT = 1
    #     gripper_value.rGTO = 1
    #     gripper_value.rATR = 0
    #     gripper_value.rPR = 255 # 爪子打开的角度, 255表示闭合
    #     gripper_value.rSP = 255
    #     gripper_value.rFR = 150
    #     self.gripper_publihser.publish(gripper_value)
    #     rospy.sleep(0.1)
    #     self.gripper_state = 1.0


    # def get_gripper_open_action(self):
    #     return 0.0
    

    # def get_gripper_close_action(self):
    #     return 1.0

    # def get_gripper_null_action(self):
    #     return self.gripper_state

    # # 初始化robotiq机械爪
    # def init_gripper(self):
    #     gripper_value = outputMsg.Robotiq2FGripper_robot_output()
    #     gripper_value.rACT = 0
    #     self.gripper_publihser.publish(gripper_value)


    # def control_open_gripper(self):
    #     print("Opening the gripper")
    #     rospy.sleep(1)
    #     gripper_value = outputMsg.Robotiq2FGripper_robot_output()
    #     gripper_value.rACT = 1
    #     gripper_value.rGTO = 1
    #     gripper_value.rSP  = 255
    #     gripper_value.rFR  = 150
    #     self.gripper_publihser.publish(gripper_value)
    #     rospy.sleep(1)


    # def control_close_gripper(self):
    #     print("Closing the gripper")
    #     rospy.sleep(1)
    #     gripper_value = outputMsg.Robotiq2FGripper_robot_output()
    #     gripper_value.rACT = 1
    #     gripper_value.rGTO = 1
    #     gripper_value.rATR = 0
    #     gripper_value.rPR = 255 # 爪子打开的角度, 255表示闭合
    #     gripper_value.rSP = 255
    #     gripper_value.rFR = 150
    #     self.gripper_publihser.publish(gripper_value)
    #     rospy.sleep(1)


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
    # def subscribe_gripper_angle_callback(self, msg: inputMsg):
    #     MAX_COUNT = 255.0
    #     OPENING_PER_COUNT_MM = 0.4  # 2F-85: 0.4mm/count
    #     gpo = float(msg.gPO)  # 0(open) .. 255(closed)
    #     self.opening_mm = (MAX_COUNT - gpo) * OPENING_PER_COUNT_MM   # 距离
    #     self.opening_pct = (MAX_COUNT - gpo) / MAX_COUNT * 100.0     # 角度
    #     self.close_num = gpo

    # def subscribr_gripper_angle(self):
    #     rospy.Subscriber("/Robotiq2FGripperRobotInput", inputMsg.Robotiq2FGripper_robot_input, 
    #                      self.subscribe_gripper_angle_callback, queue_size=10)
    #     rospy.sleep(0.2)

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



# 阈值为10mm
def detect_gripper_events_by_accumulation(coords1, coords2, threshold=10):
    """
    通过累加相邻帧距离变化量判断夹爪动作。
    
    参数:
    coords1: np.ndarray, 维度 [N, 3], 夹指1的坐标序列
    coords2: np.ndarray, 维度 [N, 3], 夹指2的坐标序列
    threshold: 累计位移阈值，单位需与输入坐标一致（如 0.01 代表 1cm）
    window_size: 平滑窗口大小，用于降低高频噪声
    
    返回:
    List[List[int, int]]: 状态变化列表 [帧索引, 状态码]
                          状态码 0: 开始闭合 (Grasp)
                          状态码 1: 开始打开 (Release)
    """
    # 计算每一帧的欧氏距离
    distances = np.linalg.norm(coords1 - coords2, axis=1)
    # 计算相邻帧的位移增量 delta_d = d_t - d_{t-1}
    deltas = np.diff(distances)
    
    events = {}
    accumulated_motion = 0.0
    start_frame = 0
    
    # 当前系统所处的状态标识：-1 寻找中, 0 正在闭合, 1 正在打开
    # 为了避免重复触发同一状态，记录上一次触发的动作类型
    last_triggered_state = -1 

    for i, delta in enumerate(deltas):
        # 如果当前的增量方向与累计方向相反，且尚未触发状态，则重新开始累计
        # 判断逻辑：如果 delta 与当前累计值异号，重置起点
        if (delta > 0 and accumulated_motion < 0) or (delta < 0 and accumulated_motion > 0):
            accumulated_motion = delta
            start_frame = i
        else:
            accumulated_motion += delta
        
        # 检查累计量是否超过阈值
        # 累计减少超过阈值 -> 判定为闭合开始
        if accumulated_motion <= -threshold:
            if last_triggered_state != 0:
                events.update({
                    start_frame: 0
                })
                last_triggered_state = 0
            # 触发后重置累计器，准备检测下一个反向动作
            accumulated_motion = 0
            
        # 累计增加超过阈值 -> 判定为打开开始
        elif accumulated_motion >= threshold:
            if last_triggered_state != 1:
                events.update({
                    start_frame: 1
                })
                last_triggered_state = 1
            # 触发后重置累计器
            accumulated_motion = 0
    return events



if __name__ == "__main__":
    
    from transform_utils import convert_pose_quat2mat, convert_pose_mat2quat
 
    T_robot_capture = np.array([[-0.02582595, -0.99962434 ,-0.0091759 ,  0.67198311],
                                [ 0.99966532, -0.02583864 , 0.00126633 , 0.40664717],
                                [-0.00150295, -0.00914013 , 0.9999571 ,  0.01686246],
                                [ 0.       ,   0.    ,      0.     ,     1.        ]])
    UMI_POS = np.array([363.4473073071311 ,1208.3283104883744, 305.06626371888166]) / 1000     
    UMITCP_POS = np.array([ 500.55924340455573 ,1214.2269269806513 ,216.94456494133658]) / 1000       

    P_UMI_UMITCP = UMITCP_POS - UMI_POS

    T_UMI_UMITCP = np.array([[1, 0, 0, P_UMI_UMITCP[0]],
                             [0, 1, 0, P_UMI_UMITCP[1]],
                             [0, 0, 1,  P_UMI_UMITCP[2]],
                             [0, 0, 0, 1]])
    
    T_capture_UMI = np.loadtxt("/home/k202/59test/000015/umi_body_abs.txt")[::10][:, 1:]  # [N 7]

    T_capture_UMI[:, :3] = T_capture_UMI[:, :3] / 1000
    T_capture_UMI = convert_pose_quat2mat(T_capture_UMI)    # [N 4 4]
    print(T_capture_UMI.shape)

    T_UMITCP_TCP = np.array([[0, 0, 1, 0],
                             [-1, 0, 0, 0],
                             [0, -1, 0, 0],
                             [0, 0, 0, 1]])
    
    T_robot_TCP = T_robot_capture[None] @ T_capture_UMI @ T_UMI_UMITCP @ T_UMITCP_TCP[None]
    T_robot_TCP = convert_pose_mat2quat(T_robot_TCP)        # 机械臂位姿
    print(T_robot_TCP)


    gripper_state = np.loadtxt("/home/k202/59test/000015/gripper_state_time.txt")
    gripper_state = gripper_state[::10][:, 1]


    rospy.init_node("UR10_Robot_Gripper_Publisher")
    tf_listener = tf.TransformListener()
    rospy.sleep(1)
    (trans, rot) = tf_listener.lookupTransform('/tool0_controller', '/tool0', rospy.Time(0))
    Ttool2tcp = np.array([trans[0], trans[1], trans[2], rot[0], rot[1], rot[2], rot[3]])
    Ttool2tcp = convert_pose_quat2mat(Ttool2tcp)
    
    robotoperation = RobotOperation(Ttool2tcp)
    # robotoperation.close_with_pos(200)
    # robotoperation.close_with_pos(0)
    # robotoperation.close_with_pos(200)
    pose = robotoperation.get_ee_pose(return_quat = True)
    
    # print(pose)# reset_reg_cost         : 1.30896
   


    # timestamps = np.loadtxt("/home/k202/lerobot/test_dk/test_arr_umi_body_abs_1.txt")[::10][: , 0:1]
    # gripper_timestamps = left_gripper_pos[:, 0]
    # num_gripper_frames = len(gripper_timestamps)
    # raw_gripper_states = np.zeros(num_gripper_frames)
    
    # current_state = 0.0
    # for i in range(num_gripper_frames):
    #     if i in gripper_event:
    #         # 状态映射：算法输出 0 为闭合，映射为 1.0；输出 1 为打开，映射为 0.0
    #         if gripper_event[i] == 0:
    #             current_state = 1.0
    #         elif gripper_event[i] == 1:
    #             current_state = 0.0
    #     raw_gripper_states[i] = current_state

    # num_frames = timestamps.shape[0]
    # gripper_states = np.zeros((num_frames, 1))
    
    # for i in range(num_frames):
    #     time_diff = np.abs(gripper_timestamps - timestamps[i, 0])
    #     closest_idx = np.argmin(time_diff)
    #     gripper_states[i, 0] = raw_gripper_states[closest_idx]

 
    # combined_data = np.hstack((timestamps, T_robot_TCP, gripper_states))
    # save_path = "/home/ywl/317data/2/predata/robot_data.txt"
    # np.savetxt(save_path,
    #            combined_data,
    #            fmt='%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %d',
    #            delimiter=' ',
    #            header='timestamp x y z qx qy qz qw gripper_state',
    #            comments='# ')

    # robotoperation.close_with_pos(100)

    
    robotoperation.UR10_moveto_pose([[-0.31895895, 0.66285471, 0.51663578, -0.93785405, -0.17891105, -0.02265816, 0.29649151]])
    T_robot_TCP = "/home/k202/test111/000004/ur10_ee_pose.txt"
    T_robot_TCP = np.loadtxt(T_robot_TCP)[:, 1:]
    gripper_state = np.loadtxt("/home/k202/test111/000004/gripper_state_record.txt")[:, 1]
    for index, i in enumerate(range(T_robot_TCP.shape[0])):
        robotoperation.UR10_moveto_pose([T_robot_TCP[i]])
        print(f"机械臂位姿：{[T_robot_TCP[i]]}")
        if gripper_state[index] == 0:
            robotoperation.close_with_pos(100)
        elif gripper_state[index] == 1:
            robotoperation.close_with_pos(0)
        else:
            print("cuowu", gripper_state[index])
    # #         # exit()

    # robotoperation.close_with_pos(40)
    # rospy.sleep(3)
    # robotoperation.close_with_pos(50)
    # rospy.sleep(3)
    # robotoperation.close_with_pos(60)
    # rospy.sleep(3)
    # robotoperation.close_with_pos(70)
    # rospy.sleep(3)
    # robotoperation.close_with_pos(0)
    # rospy.sleep(3)
    # robotoperation.close_with_pos(0)
    # # rospy.sleep(3)
    # # robotoperation.close_with_pos(0)
    # # rospy.sleep(3)
    # robotoperation.UR10_moveto_pose([[-0.31895895, 0.66285471, 0.51663578, -0.93785405, -0.17891105, -0.02265816, 0.29649151]])

