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
from scipy.spatial.transform import Rotation as R
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
from lerobot.policies.factory import make_policy,make_pre_post_processors
from lerobot.configs import parser

from UR10e_deploy.capture_rs_imgs import CameraFeedManager
from UR10e_deploy.robot_control import RobotOperation
from UR10e_deploy.transform_utils import convert_pose_quat2mat, convert_pose_euler2quat, convert_pose_euler2mat, \
    convert_pose_mat2quat
from data_trans import PI05_Data_Trans
from ur_ikfast import ur_kinematics

from dataclasses import dataclass
from lerobot.configs.policies import PreTrainedConfig
from safetensors.torch import load_file

@dataclass
class DeployConfig:
    policy: PreTrainedConfig


from pathlib import Path

class UR10ePolicyRunner:
    def __init__(self, cfg: DeployConfig):
        self.cfg = cfg
        self.max_steps = 400 

        print("Initializing UR10e Policy Runner...")
        self.device = torch.device(cfg.policy.device if cfg.policy.device else "cuda")
        
        # 1. 路径解析
        if not cfg.policy.pretrained_path:
            raise ValueError("Error: cfg.policy.pretrained_path is empty!")
        
        pretrained_path_obj = Path(cfg.policy.pretrained_path)
        if pretrained_path_obj.is_file():
            folder_path = pretrained_path_obj.parent
            weight_path = pretrained_path_obj
        else:
            folder_path = pretrained_path_obj
            weight_path = folder_path / "model.safetensors"

        print(f"Policy Folder: {folder_path}")

        # 2. 加载 Preprocessor
        print("Loading Preprocessors...")
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=str(folder_path) 
        )
        print("Preprocessors loaded.")

        # 3. 加载 Config
        print("Loading Policy Configuration...")
        # 从 json 读取网络结构
        loaded_policy_cfg = PreTrainedConfig.from_pretrained(str(folder_path))
        loaded_policy_cfg.device = cfg.policy.device
        loaded_policy_cfg.pretrained_path = str(folder_path)

        # 4. 构建 Policy (直接实例化，不再使用 make_policy)
        print("Constructing Policy Structure (Direct)...")
        self.policy = PI05Policy(loaded_policy_cfg)

        # 5. 加载权重
        if weight_path.exists():
            print(f"Loading weights from {weight_path.name}...")
            state_dict = load_file(str(weight_path))
            self.policy.load_state_dict(state_dict, strict=False)
            print("Weights loaded successfully.")
        else:
            raise FileNotFoundError(f"Weight file not found at: {weight_path}")

        # 6. 部署
        self.policy.eval()
        self.policy.to(self.device)
        print("Policy Model Ready!")

        # 7. 初始化硬件
        print("Initializing Camera...")
        self.camera_manager = CameraFeedManager()
        
        print("Initializing Robot...")
        tf_listener = tf.TransformListener()
        rospy.sleep(1)
        try:
            (trans, rot) = tf_listener.lookupTransform('/tool0_controller', '/tool0', rospy.Time(0))
        except:
            print("Warning: TF lookup failed, using identity.")
            trans, rot = [0,0,0], [0,0,0,1]

        Ttool2tcp = np.array([trans[0], trans[1], trans[2], rot[0], rot[1], rot[2], rot[3]])
        Ttool2tcp = convert_pose_quat2mat(Ttool2tcp)
        self.robot = RobotOperation(Ttool2tcp)   
        print(f"Robot Initialized. Joint Angles: {self.robot.get_joint_angle()}")
        
        self.ur10_arm_iksolver = ur_kinematics.URKinematics('ur10')
#############################################################################################################################################

    def get_observation(self, task_intru):
        """
        获取观测数据，并严格对齐 /media/ywl/T7 Shield/lerobot_dataset_converted/pick_and_place/meta/info.json 定义的训练格式
        Target Keys: ['observation.images.cam_wrist', 'observation.state', 'task']
        """
        print("\n[DEBUG] 1. 进入 get_observation...")
        # 1. 采集原始数据
        # --- 诊断点 1: 机械臂通信 ---
        print("[DEBUG] 2. 正在请求机械臂位姿 (get_ee_pose)...")
        quat_state = self.robot.get_ee_pose(return_quat=True) 
        print(f"[DEBUG] >> 成功获取位姿: {quat_state}")

        # --- 诊断点 2: 相机通信 ---
        print("[DEBUG] 3. 正在请求相机图像 (get_latest_frames)...")
        _, _, cam_high, _ = self.camera_manager.get_latest_frames()
        print("[DEBUG] >> 成功获取图像")

        # 2. 格式化：对齐 observation.state
        self.robot.subscribr_gripper_angle()
        gripper_state = self.robot.close_num / 255.0
        # 拼接成 8 维向量
        state_np = np.append(quat_state, gripper_state)
        state_tensor = torch.from_numpy(state_np).float()

        # 3. 格式化：对齐 observation.images.cam_wrist (Shape: [3, 480, 640])
        cam_high_tensor = torch.from_numpy(cam_high).float() / 255.0
        cam_high_tensor = cam_high_tensor.permute(2, 0, 1) # [H,W,C] -> [C,H,W]

        # 4. 返回符合 LeRobotDataset 标准的字典
        return {
            #########pi0.5keys值
            #"observation.images.right_wrist_0_rgb": cam_high_tensor,
            "observation.images.cam_high": cam_high_tensor,
            "observation.state": state_tensor,
            "task": task_intru  
        }
######################################################################################################################################################


    # 逆运动学求解关节角
    def ur_iksolver(self, target_pose_quat):
        self.robot.get_joint_angle()
        ik_solver_res = self.ur10_arm_iksolver.inverse(target_pose_quat, 
                                     False,
                                     q_guess = self.robot.joint_angle)  # 当前关节角作为优化初值
        
        return ik_solver_res
        
    def run(self, task_intru):
        print(f"Starting policy execution loop for task: {task_intru}")
        
        step = 0
        gripper_all = []
        self.policy.config.n_action_steps = 2
        while step < self.max_steps:
            if len(self.policy._action_queue) == 0:
                # 1. 获取观测 and 预处理
                raw_obs = self.get_observation(task_intru)
                print(f'raw_obs.keys:{raw_obs.keys()}')
                batch = self.preprocessor(raw_obs)
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device, non_blocking=True)

            action_num = 1
            action_pose_list = []
            # 2. 模型推理 
            with torch.inference_mode():
                # 获取原始动作
                while len(action_pose_list) < action_num:
                    action = self.policy.select_action(batch)
                    action = self.postprocessor(action)
                    action_pose_list.append(action) # [[1 8]shape, [1 8]shape, ...]
                    gripper_all.append(action[0, -1] * 255)
                
                np.savetxt("/home/ywl/lerobot/UR10e_deploy/gripper_value.txt", np.array(gripper_all))
                start_pose = np.eye(4)  # [4 4]
                start_gripper = 0
                for i in range(action_num):
                    current_all = action_pose_list[i]
                    current_pose_quat = current_all[:, :7]   # [1 7]
                    current_gripper = current_all[:, [-1]]  # [1 1]
                    current_pose_mat = convert_pose_quat2mat(current_pose_quat.cpu().numpy())[0]   # [4 4]
                    start_pose = np.matmul(start_pose, current_pose_mat)    # [4 4]
                    start_gripper += current_gripper.cpu().numpy()[0, 0]

                start_pose = torch.from_numpy(convert_pose_mat2quat(start_pose[None])).cuda()
                start_gripper = torch.from_numpy(start_gripper[None, None]).cuda()
                action = torch.cat((start_pose, start_gripper), dim=1)


            print("预测位姿", action)
            # 3. 动作解析 
            action_numpy = action.cpu().numpy()
            pose_quat_numpy = action_numpy[:, :7]
            pose_mat_numpy = convert_pose_quat2mat(pose_quat_numpy)[0]    # [4 4]

            self.robot.subscribr_gripper_angle()
            gripper_val = self.robot.close_num

            gripper_dis_val = action_numpy[:, -1]
            gripper_dis_val = gripper_dis_val * 255.0
            gripper_val = gripper_val + gripper_dis_val
            gripper_val = max(0.0, min(255.0, gripper_val))

            curr_pose = self.robot.get_ee_pose(return_quat=False)   # [4 4] 
            action_mat_pose = np.matmul(curr_pose, pose_mat_numpy)
            action_quat_pose = convert_pose_mat2quat(action_mat_pose)
            print(gripper_val)

            ############################################33
            #self.robot.close_gripper_num(gripper_val)

            if gripper_dis_val >=20 or gripper_dis_val <= -3:
                gripper_val = 255 if gripper_dis_val >=20 else 0
                self.robot.close_gripper_num(gripper_val)
            # self.robot.close_gripper_num(min(max(0, gripper_val), 255))
            self.robot.UR10_moveto_pose(list(action_quat_pose[None]))


           #robotoperation.UR10_moveto_pose([[-0.31895895, 0.66285471, 0.51663578, -0.93785405, -0.17891105, -0.02265816, 0.29649151]])

            # delta_pos = action_numpy[:3]      # 位置增量
            # delta_rot_vec = action_numpy[3:6] # 旋转增量 (轴角)
            # gripper_action = action_numpy[6]

            # curr_pose = self.robot.get_ee_pose(return_quat=False)   # [4 4] 

            
            # # 4. 积分计算- 算出绝对目标
           
            # # 获取当前绝对位姿 
            # curr_pose = self.robot.get_ee_pose(return_quat=True) 
            # curr_pos = curr_pose[:3]
            # curr_quat = curr_pose[3:7]

            # # A. 位置积分
            # scale_pos = 1.0 
            # target_pos = curr_pos + (delta_pos * scale_pos)

            # # B. 姿态积分 (必须在四元数空间做，否则增量会错)
            # r_curr = R.from_quat(curr_quat)
            # r_delta = R.from_rotvec(delta_rot_vec)
            # r_target = r_curr * r_delta  # 旋转叠加


            # # 5. 适配原接口 
   
            # # 在内部用了四元数计算，但为了配合旧代码，
            # # 将计算好的【绝对目标姿态】转回【欧拉角】
            
            # target_euler = r_target.as_euler('xyz', degrees=False)

            # # 组合成 6维向量 [x, y, z, roll, pitch, yaw]
            # # 原函数想要的 "pose_action" 格式，但现在它是绝对目标，不再是增量
            # target_pose_6d = np.concatenate([target_pos, target_euler])

            # gripper_val = int(gripper_action * 255)
            # gripper_val = max(0, min(255, gripper_val))

            # move_dist = np.linalg.norm(delta_pos) # 移动距离 (米)
            # rot_angle = np.linalg.norm(delta_rot_vec) # 转动角度 (弧度)
            # print("-" * 30)
            # print(f"Step {step} 决策分析:")
            # print(f"  [1] 模型预测增量: 移动 {move_dist*1000:.1f} mm, 转动 {np.degrees(rot_angle):.1f} 度")
            # print(f"  [2] 目标绝对位置: {target_pos}")
            # print(f"  [3] 目标绝对欧拉: {target_euler}") # 方便你直观检查角度是否正常
            
            
            
            # # 只要这个变量是 True，机械臂就绝对不会动
            # DRY_RUN_MODE = True  
            
            # if DRY_RUN_MODE:
            #     print("【安全模式】指令已计算，但未发送给硬件。")
            #     time.sleep(0.5) 
            # else:
            #     # 上面改为 False 时，才会真的动
            #     print("【警告】正在执行物理运动！！！")
            #     self.robot.close_gripper_num(gripper_val)
            #     self.robot.UR10_moveto_pose(convert_pose_euler2quat(target_pose_6d))

            step += 1



if __name__ == "__main__":
    rospy.init_node("UR10_PI05")
    
    @parser.wrap()
    def eval_main(cfg: DeployConfig): 
        return cfg
    
    cfg = eval_main()

    UR10_runner = UR10ePolicyRunner(cfg)

    UR10_runner.run(task_intru="pick up the bottle and place it in the box")


