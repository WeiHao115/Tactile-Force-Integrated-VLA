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
        # rospy.init_node("UR10_Robot_Gripper_Publisher")
        self.trajectory_publihser = rospy.Publisher('/scaled_pos_joint_traj_controller/command', JointTrajectory, queue_size=10)
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


    # def UR10_moveto_IKSolver(self, opt_pose_quat):
    #     moveit_commander.roscpp_initialize(sys.argv)
    #     arm = moveit_commander.MoveGroupCommander("manipulator")

    #     basetobaselink = np.array([[-1, 0, 0, 0],
    #                                [0, -1, 0, 0],
    #                                [0, 0, 1, 0],
    #                                [0, 0, 0, 1]])
    #     base_link_opt_pose_homo = np.matmul(basetobaselink, convert_pose_quat2mat(np.array(opt_pose_quat)))
    #     self.get_joint_angle()
    #     reset_joint_pos = self.joint_angle
    #     # IK Solver求解结果
    #     ik_solver_res = self.ur10_arm.inverse(convert_pose_mat2quat(base_link_opt_pose_homo), 
    #                                     False,
    #                                     q_guess = reset_joint_pos)
    #     self.UR10_moveto_angle(ik_solver_res)

    

    
    #-------------------------------------------------------------------------------------------
    # TODO by DK -> MODIFIED by Gemini
    # 设置机械臂各个关键的初始角
    def reset_joint_pos(self, duration_sec=5.0):
        """
        将机械臂移动到一个预定义的、安全的 "Home" 位置。
        使用 UR10_movetset_traceo_angle 方法执行。
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
            0.0                     # wrist_3_joint
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
        now = rospy.Time.now()
        self.tf_listener.waitForTransform('/base', '/tool0_controller', now, rospy.Duration(0.5))
        (trans, rot) = self.tf_listener.lookupTransform('/base', '/tool0_controller', rospy.Time(0))
        pose_numpy = np.array([trans[0], trans[1], trans[2], rot[0], rot[1], rot[2], rot[3]])
        pose_matrix = convert_pose_quat2mat(pose_numpy)
        if return_quat:
            return pose_numpy
        return pose_matrix  # [4 4]
    


    def get_ee_pose_moveit(self, return_quat=False):
        current_pose_msg = self.move_group.get_current_pose().pose

        target_position = np.array([current_pose_msg.position.x, current_pose_msg.position.y, current_pose_msg.position.z,
                                    current_pose_msg.orientation.x, current_pose_msg.orientation.y, current_pose_msg.orientation.z,
                                    current_pose_msg.orientation.w])
        T_bl_falan = convert_pose_quat2mat(target_position)
        T_b_bl = np.linalg.inv(np.array([[-1, 0, 0, 0],
                                        [0, -1, 0, 0],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]]))
        T_b_falan = T_b_bl @ T_bl_falan
        T_falan_tcp = np.linalg.inv(self.Ttool2tcp)
        pose_numpy = convert_pose_mat2quat(T_b_falan @ T_falan_tcp)
        # pose_numpy[3:] = pose_numpy[3:] * -1
        if return_quat:
            return pose_numpy
        pose_matrix = convert_pose_quat2mat(pose_numpy)
        return pose_matrix




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
    
    def init_gripper(self):
        
        self.MAX_REGISTER = 1000.0
        self.MAX_STROKE_MM = 200.0
        self.current_pos_register = -1
        self.opening_mm = self.MAX_STROKE_MM
        self.opening_pct = 0.0
        self.close_num = 0.0
        self.gripper_state = 0.0

        #rospy.init_node('dh_gripper_python_client', anonymous=True)
        self.pub_force = rospy.Publisher('/gripper/close_with_force', Float32, queue_size=10)
        self.pub_pos_mm = rospy.Publisher('/gripper/set_pos_mm', Float32, queue_size=10)
        self.sub_status = rospy.Subscriber('/gripper/curr_pos', Int32, self.subscribr_gripper_angle)

        rospy.sleep(1.0)
        rospy.loginfo("DH Gripper Client Initialized.")

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
        msg = Float32()
        msg.data = float(target_pos_mm)
        self.pub_pos_mm.publish(msg)

    def close_gripper_num(self, clouse_num):
        clouse_num = max(0.0, min(float(clouse_num), 100.0))
        if self.gripper_state == 0.0 and clouse_num > 1:
            clouse_num = 100.0
            self.gripper_state = 1.0
        elif self.gripper_state == 0.0 and clouse_num < 95:
            clouse_num = 0.0
            self.gripper_state = 0.0
        elif self.gripper_state == 1.0 and clouse_num < 1:
            clouse_num = 0.0
            self.gripper_state = 0.0
        else:
            clouse_num = 100.0
            self.gripper_state = 1.0


        target_mm = self.MAX_STROKE_MM * (1.0 - (clouse_num / 100.0))
        msg = Float32()
        msg.data = target_mm
        self.pub_pos_mm.publish(msg)
        
        rospy.sleep(0.25)



if __name__ == "__main__":
    rospy.init_node("UR10_Robot_Gripper_Publisher")
    tf_listener = tf.TransformListener()
    rospy.sleep(1)
    (trans, rot) = tf_listener.lookupTransform('/tool0_controller', '/tool0', rospy.Time(0))
    Ttool2tcp = np.array([trans[0], trans[1], trans[2], rot[0], rot[1], rot[2], rot[3]])
    Ttool2tcp = convert_pose_quat2mat(Ttool2tcp)
    
    robotoperation = RobotOperation(Ttool2tcp)
    
    import time
    start_time = time.time()
    pose = robotoperation.get_ee_pose(return_quat = True)
    end_time = time.time()
    print(end_time - start_time)
    print(pose)

    start_time = time.time()
    pose = robotoperation.get_ee_pose_moveit(return_quat = True)
    end_time = time.time()
    print(end_time - start_time)
    print(pose)
    # exit()

    #print(pose)# reset_reg_cost         : 1.30896
    # exit()
    # rospy.sleep(5)
    # print("夹爪闭合50%")
    # robotoperation.close_gripper_num(50)
    # rospy.sleep(5)
    # print("夹爪闭合100%")
    # robotoperation.close_gripper_num(100)
    # rospy.sleep(5)
    # print("夹爪打开")
    robotoperation.close_gripper_num(100)
    rospy.sleep(1)
    robotoperation.close_gripper_num(0)
    rospy.sleep(1)
    # ！！！！！！！！！！！！！BEST POSE
    #robotoperation.UR10_moveto_pose([[-0.31895895, 0.66285471, 0.51663578, -0.93785405, -0.17891105, -0.02265816, 0.29649151]])
  
    #UMI起始位姿
    # robotoperation.UR10_moveto_pose([[-0.449306, 0.755043, 0.124363, -0.997934, -0.037414, 0.049210, 0.017526]])
    robotoperation.UR10_moveto_pose([[-0.431347,  0.724575,  0.138996, -0.923718, -0.025503,  0.041791,  0.379931]])

    # robotoperation.UR10_moveto_pose([[-0.4329,  0.6481,  0.4966, -0.9341, -0.1634, -0.0201,  0.3169]])
    # robotoperation.UR10_moveto_pose([[-0.4397,  0.7502,  0.1838,  0.9826, -0.0101, -0.0814, -0.1665]])
    #robotoperation.UR10_moveto_pose([[-0.5086,  0.4094,  0.4879,  0.0859,  0.3307,  0.9380,  0.0578]])
    #robotoperation.UR10_moveto_pose([[  -0.4659,  0.7025,  0.0628,  0.9274, -0.0289, -0.0249, -0.3721]])
