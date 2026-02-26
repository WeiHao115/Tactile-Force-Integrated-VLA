import sys
import os
import pickle
import gc
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm

# === 配置路径 ===
LEROBOT_PATH = "/home/ywl/lerobot/src" 
if LEROBOT_PATH not in sys.path:
    sys.path.append(LEROBOT_PATH)

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.video_utils import VideoEncodingManager
    print("成功导入: lerobot.datasets.lerobot_dataset")
except ImportError as e:
    print(f"导入 LeRobot 失败: {e}")
    exit(1)

# ==========================================
# 用户配置区
# ==========================================
# 1. 任务名称
TASK_NAME = "pick_and_place"

# 2. 任务描述
TASK_DESC = "pick up the bottle and place it in the box"

# 3. 路径配置
RAW_DATA_DIR = "/media/ywl/T7 Shield/lerobot_raw_data/1"
BASE_OUTPUT_DIR = "/media/ywl/T7 Shield/lerobot_dataset_converted11"

OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, TASK_NAME)
REPO_ID = f"user/{TASK_NAME}"

FPS = 30                        
ROBOT_TYPE = "ur10" # LeRobot 建议用小写
JOINT_DIM = 6       # UR10 关节数
# ==========================================

def get_image_size(episode_data):
    """从第一帧获取图像分辨率，并检查有哪些相机"""
    first_frame = episode_data[0]
    
    # 检查有哪些相机数据
    has_high = "observation.images.cam_high" in first_frame and first_frame["observation.images.cam_high"] is not None
    has_wrist = "observation.images.cam_wrist" in first_frame and first_frame["observation.images.cam_wrist"] is not None
    
    img = None
    if has_high:
        img = first_frame["observation.images.cam_high"]
    elif has_wrist:
        img = first_frame["observation.images.cam_wrist"]
    else:
        raise ValueError(f"无法找到图像数据, Available Keys: {first_frame.keys()}")
    
    return img.shape[0], img.shape[1], has_high, has_wrist

def convert_to_v3():
    pkl_paths = sorted(list(Path(RAW_DATA_DIR).glob("*.pkl")))
    if not pkl_paths:
        print(f"Error: 目录 {RAW_DATA_DIR} 下没有找到 .pkl 文件")
        return
    print(f"扫描到 {len(pkl_paths)} 个数据文件。准备开始转换...")

    # 2. 读取第一个文件以确定配置
    with open(pkl_paths[0], "rb") as f:
        print("开始读取pkl文件")
        first_ep_data = pickle.load(f)
    img_h, img_w, has_high, has_wrist = get_image_size(first_ep_data)
    print(f"检测到图像分辨率: {img_w}x{img_h}")
    print(f"相机状态: High={'✅' if has_high else '❌'}, Wrist={'✅' if has_wrist else '❌'}")
    
    del first_ep_data
    # gc.collect()

    # 3. 定义数据集特征 (Feature Definition)
    # Copy from hw_to_dataset_features
    # /home/ywl/lerobot/src/lerobot/datasets/utils.py
    features = {
        # # --- 状态输入 (Proprioception) ---
        # "observation.joints": {
        #     "dtype": "float32",
        #     "shape": (JOINT_DIM,),
        #     "names": ["j1", "j2", "j3", "j4", "j5", "j6"],
        # },
        "observation.state": {
            "dtype": "float32",
            "shape": (8,),
            "names": ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"],
        },
        "action": {
            "dtype": "float32",
            "shape": (7,), 
            "names": ["x", "y", "z", "qx", "qy", "qz", "qw"],
        },
    }

    # --- 动态添加相机特征 ---
    if has_high:
        features["observation.images.cam_high"] = {
            "dtype": "video",
            "shape": (img_h, img_w, 3),
            "names": ["height", "width", "channels"],
        }
    if has_wrist:
        features["observation.images.cam_wrist"] = {
            "dtype": "video",
            "shape": (img_h, img_w, 3),
            "names": ["height", "width", "channels"],
        }


    if os.path.exists(OUTPUT_DIR):
        print(f"清理旧目录: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        fps=FPS,
        root=OUTPUT_DIR,
        features=features,
        use_videos=True,
        robot_type=ROBOT_TYPE,
        image_writer_threads=4,
        batch_encoding_size=1
    )

    # 5. 逐个文件处理
    total_frames = 0
    ignored_episodes = 0
    with VideoEncodingManager(dataset):
        for ep_idx, pkl_path in enumerate(pkl_paths):
            print(f"[{ep_idx+1}/{len(pkl_paths)}] 处理文件: {pkl_path.name}")
            try:
                with open(pkl_path, "rb") as f:
                    episode_data = pickle.load(f)
            except Exception as e:
                print(f"读取失败 {pkl_path}: {e}")
                continue
                
            if len(episode_data) < 2:
                print(f"数据太短，跳过。")
                ignored_episodes += 1
                continue

            # 遍历帧
            for frame in tqdm(episode_data, desc=f"Ep {ep_idx}", leave=False):
                frame["task"] = TASK_DESC
                dataset.add_frame(frame)
                total_frames += 1

            dataset.save_episode()

            # 内存回收
            # del episode_data
            # gc.collect()

    dataset.finalize()
    print(f"转换完成！数据集已保存至: {OUTPUT_DIR}")
    print(f"总帧数: {total_frames}, 忽略的短片段: {ignored_episodes}")

if __name__ == "__main__":
    convert_to_v3()