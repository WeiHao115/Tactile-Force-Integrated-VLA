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
sys.path.append("/home/k202/lerobot/src")
sys.path.append("/home/k202/lerobot")

# LeRobot imports
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.utils.control_utils import predict_action
from lerobot.utils.utils import get_safe_torch_device
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.policies.factory import make_policy,make_pre_post_processors
from lerobot.configs import parser
from lerobot.datasets.factory import make_dataset

from UR10e_deploy.gopro_tac_reader import GelSightManager,GoproManager,RealsenseRosManager
from UR10e_deploy.robot_control import RobotOperation
from UR10e_deploy.transform_utils import convert_pose_quat2mat, convert_pose_euler2quat, convert_pose_euler2mat, \
    convert_pose_mat2quat
from data_trans import PI05_Data_Trans


from dataclasses import dataclass
from lerobot.configs.policies import PreTrainedConfig
from safetensors.torch import load_file

@dataclass
class DeployConfig:
    policy: PreTrainedConfig


from pathlib import Path

def process_and_resize_frame(frame_bgr, target_size):
    """
    对内存中的 NumPy 图像阵列进行中心裁剪和缩放
    """
    if frame_bgr is None:
        return None
        
    h, w = frame_bgr.shape[:2]
    short_edge = min(h, w)
    start_x = (w - short_edge) // 2
    start_y = (h - short_edge) // 2
    
    img_cropped = frame_bgr[start_y:start_y+short_edge, start_x:start_x+short_edge]
    img_resized = cv2.resize(img_cropped, target_size)
    
    return img_resized

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


        print("加载数据集并获得统计数据中...")
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
        #self.tactile_manager = GelSightManager()
        self.gopro_manager = GoproManager(device_id=6, width=224, height=224, fps=30)
        self.realsense = RealsenseRosManager()


        
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
    

    def get_observation(self, task_intru):
        quat_state = self.robot.get_ee_pose(return_quat=True) 
        gopro_image, _ = self.gopro_manager.get_latest_frame()
        realsense_image , _ =self.realsense.get_latest_frame()

        # if gopro_image is not None :
        #     cv2.imshow("GoPro Camera", gopro_image)
        #     cv2.waitKey(50)
        # if gopro_image is not None :
        #     cv2.imshow("Realsense Camera", realsense_image)
        #     cv2.waitKey(50)

        # print("4. 正在请求触觉图像 ...")
        # tactile_image_1, tactile_image_2, _= self.tactile_manager.get_tactile_frame()
        # print("成功获取触觉图像")
    
        gripper_state = self.robot.close_num 
        gripper_state = 1 if gripper_state / 100 > 0.25 else 0
        state_np = np.append(quat_state, gripper_state)
        state_tensor = torch.from_numpy(state_np).float()

        gopro_image = cv2.cvtColor(gopro_image, cv2.COLOR_BGR2RGB)
        gopro_image_tensor = torch.from_numpy(gopro_image).float() / 255.0
        gopro_image_tensor = gopro_image_tensor.permute(2, 0, 1) # [H,W,C] -> [C,H,W]

        realsense_image = cv2.cvtColor(realsense_image, cv2.COLOR_BGR2RGB)
        realsense_image_tensor = torch.from_numpy(realsense_image).float() / 255.0
        realsense_image_tensor = realsense_image_tensor.permute(2, 0, 1) # [H,W,C] -> [C,H,W]


        # tactile_image_1_tensor = torch.from_numpy(tactile_image_1).float() / 255.0
        # tactile_image_1_tensor = tactile_image_1_tensor.permute(2,0,1)
        # tactile_image_2_tensor = torch.from_numpy(tactile_image_2).float() / 255.0
        # tactile_image_2_tensor = tactile_image_2_tensor.permute(2,0,1)

        # 4. 返回符合 LeRobotDataset 标准的字典
        return {
            "observation.images.gopro": gopro_image_tensor,
            "observation.images.realsense": realsense_image_tensor,
            # "observation.images.tactile_left": tactile_image_1_tensor,
            # "observation.images.tactile_right": tactile_image_2_tensor,
            "observation.state": state_tensor,
            "task": task_intru  
        }

        
    def run(self, task_intru):
        print(f"Starting policy execution loop for task: {task_intru}")
        step = 0
        self.policy.config.n_action_steps = 1
        while step < self.max_steps:
            if len(self.policy._action_queue) == 0:
                raw_obs = self.get_observation(task_intru)
                print(f'raw_obs.keys:{raw_obs.keys()}')
                # tensor([[-0.4701,  0.7279,  0.1794, -0.9925, -0.0394,  0.0687,  0.0934,  0.0000]])
                batch = self.preprocessor(raw_obs)
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device, non_blocking=True)

            action_num = 10
            action_pose_list = []
            gripper_all = []
            # 2. 模型推理 
            # self.robot.close_gripper_num(100)
            # rospy.sleep(5)
            with torch.inference_mode():
                # 获取原始动作
                while len(action_pose_list) < action_num:
                    # import pdb; pdb.set_trace()
                    # plt.imsave("/home/k202/lerobot/src/lerobot/scripts/train_gopro_wtq.jpg", (dataset[10]['observation.images.gopro'].permute(1, 2, 0).cpu().numpy()))
                    # cv2.imwrite("/home/k202/lerobot/src/lerobot/scripts/train_gopro_wtq.jpg", (dataset[0]['observation.images.gopro']*255).permute(1,2,0).cpu().numpy().astype(np.int32))
                    # cv2.imwrite("/home/k202/lerobot/src/lerobot/scripts/test_gopro.jpg", (batch['observation.images.gopro'][0] * 255).permute(1,2,0).cpu().numpy().astype(np.int32))
                    # cv2.imwrite("/home/k202/lerobot/src/lerobot/scripts/test_realsense.jpg", (batch['observation.images.realsense'][0] * 255).permute(1,2,0).cpu().numpy().astype(np.int32))
                    # exit()
                    action = self.policy.select_action(batch)
                    action = self.postprocessor(action)
                    action_pose_list.append(action) # [[1 8]shape, [1 8]shape, ...]
                    gripper_all.append(action[0, -1])

                gripper_all_mask_index = np.where(np.array(gripper_all) < 0.1)
                if (np.array(gripper_all) < 10).sum() / action_num <= 0.1:
                    for sing_index in gripper_all_mask_index:
                        gripper_all[sing_index] = 0.9

                for i in range(action_num):
                    current_all = action_pose_list[i]
                    current_pose_quat = current_all[:, :7]   # [1 7]
                    current_gripper = current_all[:, -1]  # [1]
    
                    current_pose_mat = convert_pose_quat2mat(current_pose_quat.cpu().numpy())[0]   # [4 4]
                    curr_pose = self.robot.get_ee_pose_moveit(return_quat=False)   # [4 4] 
                    action_mat_pose = np.matmul(curr_pose, current_pose_mat)
                    action_quat_pose = convert_pose_mat2quat(action_mat_pose)

                    current_gripper = current_gripper.item()

                    target_gripper_val = current_gripper * 100.0
                    gripper_val = max(0.0, min(100.0, target_gripper_val))
                    print(f"目标夹爪闭合百分比: {gripper_val}")
                    # self.robot.close_gripper_num(gripper_val)
                    print("预测运动位姿:", current_pose_quat)
                    print("===============================================")
                    self.robot.UR10_moveto_pose(list(action_quat_pose[None]))

            step += 1



if __name__ == "__main__":
    
    try:
        rospy.init_node("UR10_PI05")
        
        @parser.wrap()
        def eval_main(cfg: DeployConfig): 
            return cfg
        
        cfg = eval_main()
        UR10_runner = UR10ePolicyRunner(cfg)
        UR10_runner.run(task_intru="Insert the plug into the power strip")


    except KeyboardInterrupt:
        print("\n检测到键盘中断 (Ctrl+C)。准备强制终止当前任务...")
        
    except Exception as e:
        print(f"\n运行时发生异常: {e}")
        
    finally:
        print("\n开始执行安全退出流程并释放硬件资源...")
        if UR10_runner is not None:
            if hasattr(UR10_runner, 'gopro_manager') and UR10_runner.gopro_manager is not None:
                try:
                    UR10_runner.gopro_manager.release()
                    print("GoPro 相机资源已释放。")
                except Exception as e:
                    print(f"释放 GoPro 时发生异常: {e}")
            
            if hasattr(UR10_runner, 'tactile_manager') and UR10_runner.tactile_manager is not None:
                try:
                    UR10_runner.tactile_manager.release()
                    print("触觉传感器资源已释放。")
                except Exception as e:
                    print(f"释放触觉传感器时发生异常: {e}")

        cv2.destroyAllWindows()
        print("所有硬件及系统资源清理完毕，进程安全退出。")

#查看端口v4l2-ctl --list-devices



