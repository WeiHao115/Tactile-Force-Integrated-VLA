'''
    1、通过观测数据, 使用pi0.5生成末端位姿, 控制机械臂运动
    2、self.policy = make_policy(
            cfg = cfg.policy, env_cfg = cfg.env, rename_map=cfg.rename_map
        )
        如上代码所示, 在__init__函数中定义pi0.5时, 我参考libero指定了按照cfg.env来定义输入是什么输出是什么
        但/home/ywl/lerobot/src/lerobot/policies/factory.py第399行定义中, 对于我们自己采集的数据
        我感觉应该使用ds_meta参数(LeRobotDatasetMetadata类), 
        所以需要**了解怎么把我们自己采集的数据转换为LeRobotDatasetMetadata类**
    
    3、整体流程在 def run函数中定义
        (1) self.get_observation返回观测值, 包括视觉图像与机械臂自身状态
        (2) self.data_trans_func将观测数据转换为pi0.5输入格式
            和/home/ywl/lerobot/src/lerobot/scripts/lerobot_eval.py函数中第226行的observation对齐
        (3) self.policy.select_action前向推理生成机器人动作
        (4) 控制机器人本体和夹爪运动
    4、**144行中的import pdb代码不要解开, 否则机械臂会一直连续运动, 有点危险**
    5、108行和132行的数据格式, 我也是按照libero定义的
    5、**后面要做的事**
        (1) 了解怎么把我们自己采集的数据转换为LeRobotDatasetMetadata类, 修改代码, 使其适配自己采集的数据
                我个人感觉可以按照libero的格式来定义自己采集的数据
        (2) 我写的只是推理代码, 要把(自己采集数据->转换为LeRobotDatasetMetadata类->训练pi0.5这一流程打通)
'''

import numpy as np
import cv2
import time
import threading
import rospy
from pathlib import Path
import torch
import tf
import sys
sys.path.append("/home/ywl/lerobot/src")
sys.path.append("/home/ywl/lerobot")

# LeRobot imports
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.utils.control_utils import predict_action
from lerobot.utils.utils import get_safe_torch_device
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.policies.factory import make_policy
from lerobot.configs import parser

from UR10e_deploy.capture_rs_imgs import CameraFeedManager
from UR10e_deploy.robot_control import RobotOperation
from UR10e_deploy.transform_utils import convert_pose_quat2mat, convert_pose_euler2quat
from data_trans import PI05_Data_Trans
from ur_ikfast import ur_kinematics

class UR10ePolicyRunner:
    # policy_path: 预训练模型权重
    def __init__(self, cfg: EvalPipelineConfig):
        self.cfg = cfg
        self.max_steps = 400 # 最大执行步数

        print("Initializing UR10e Policy Runner...")
        self.device = cfg.policy.device
        # Modified by DK, 真实场景中这里要怎么去定义？
        # 这里会加载预训练权重, 此外会在cfg中指定输入是什么输出是什么
        # 自己部署的话env_cfg要改为ds_meta，ds_meta是一个LeRobotDatasetMetadata类，.features定义了数据属性
        # 这一块要去调研一下怎么将自己的数据定义为LeRobotDatasetMetadata的格式
        self.policy = make_policy(
            cfg = cfg.policy, env_cfg = cfg.env, rename_map=cfg.rename_map
        )
        self.policy.eval()
        self.policy.reset() # 清空缓存
        print("Policy Loaded Successfully")

        # Initialize camera
        print("Initializing Camera...")
        self.camera_manager = CameraFeedManager()
        print("Camera Initialed Finished...")
        
        # Initialize robot connection
        print("Establishing the Robot Controller...")
        tf_listener = tf.TransformListener()
        rospy.sleep(1)
        (trans, rot) = tf_listener.lookupTransform('/tool0_controller', '/tool0', rospy.Time(0))
        Ttool2tcp = np.array([trans[0], trans[1], trans[2], rot[0], rot[1], rot[2], rot[3]])
        Ttool2tcp = convert_pose_quat2mat(Ttool2tcp)
        self.robot = RobotOperation(Ttool2tcp)   # 初始化机器人控制函数
        print(f"机器人初始位姿为(XYZ+四元数): {self.robot.get_ee_pose(return_quat = True)}")
        self.robot.get_joint_angle()
        print(f"机器人关节角为: {self.robot.joint_angle}")
        print("Establishing the Robot Controller Finished...")
        
        # Transforming the Data Format
        print("Establishing the Data Format Fuunction...")
        self.data_trans_func = PI05_Data_Trans()
        print("Establishing the Data Format Fuunction Finished...")

        self.ur10_arm_iksolver = ur_kinematics.URKinematics('ur10')


    def get_observation(self, task_intru):
        """Get current observation from robot and cameras"""
        quat_state = self.robot.get_ee_pose(return_quat = True) # XYZ+四元数 [7]
        # Modified by DK
        #rgb, depth = self.camera_manager.get_latest_frames() 

        # 接收四个参数  Modified by Weihao
        ext_rgb, ext_dep, wrist_rgb, wrist_dep = self.camera_manager.get_latest_frames()
        self.robot.get_joint_angle()
        self.robot.subscribr_gripper_angle()

        quat_state = torch.from_numpy(quat_state)[None] # [1 7]
        # Modified by DK
        # 这里适合libero对齐的，后续要针对实际数据进行修改
        robot_gripper = torch.tensor([[self.robot.close_num / 255.0, -self.robot.close_num / 255.0]])
        return {
            "observation.image": [ext_rgb, wrist_rgb],
            "observation.robot_state.eef.mat": convert_pose_quat2mat(quat_state),
            "observation.robot_state.eef.pos": quat_state[:, :3],
            "observation.robot_state.eef.quat": quat_state[:, 3:],
            "observation.robot_state.gripper.qpos": robot_gripper,  # Modified by DK, 机械臂关闭程度，0表示完全打开，255表示完全关闭
            "observation.robot_state.joints.pos": self.robot.joint_angle,
            "task": task_intru
        }

    # 逆运动学求解关节角
    def ur_iksolver(self, target_pose_quat):
        self.robot.get_joint_angle()
        ik_solver_res = self.ur10_arm_iksolver.inverse(target_pose_quat, 
                                     False,
                                     q_guess = self.robot.joint_angle)  # 当前关节角作为优化初值
        
        return ik_solver_res
        

    def run(self, task_intru):
        print("Starting policy execution loop...")

        observation = self.get_observation(task_intru)
        observation = self.data_trans_func(self.cfg, 
                                           observation["task"],
                                           observation["observation.image"],
                                           observation["observation.robot_state.eef.mat"],
                                           observation["observation.robot_state.eef.pos"],
                                           observation["observation.robot_state.eef.quat"],
                                           observation["observation.robot_state.gripper.qpos"],
                                           observation["observation.robot_state.joints.pos"])
        step = 0
        while step < self.max_steps:
            loop_start_time = time.time()
            with torch.inference_mode():
                action_values = self.policy.select_action(observation)  # [1 7]
                action_values = action_values.cpu().numpy()


                print(f"Top cam: {observation['observation.images.image'].shape}, Wrist cam: {observation['observation.images.image2'].shape}")


            # 现在运动轨迹不好，先不要解开
            import pdb; pdb.set_trace()
            self.robot.close_gripper_num((action_values[0, 6] * 255).astype(np.int16))
            self.robot.UR10_moveto_pose(convert_pose_euler2quat(action_values[:, :6]))
            step += 1


if __name__ == "__main__":
    rospy.init_node("UR10_PI05")
    @parser.wrap()
    def eval_main(cfg: EvalPipelineConfig):
        return cfg
    cfg = eval_main()

    UR10_runner = UR10ePolicyRunner(cfg)
    UR10_runner.run(task_intru = "pick up the bottle and place it in the box")



