import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

repo_id = "/home/k202/Insert_Notac_0503_ur10/Insert_plug"
target_episode_index = 0
dataset = LeRobotDataset(
    repo_id="Insert_plug",  # 给数据集起个名字
    root=repo_id,        # 明确指向数据集所在的父目录或根目录
    episodes=[target_episode_index]
)

# 1. 在加载时直接指定需要的 episode 索引
# 此时 dataset 内部将被重组，只加载并包含第 10 个 episode 的数据
dataset = LeRobotDataset(repo_id, episodes=[target_episode_index])

# 2. 由于数据集已被过滤，局部索引 0 即代表该 episode 的第一帧
first_frame_data = dataset[45]

# 3. 提取位姿
first_frame_pose = first_frame_data["observation.state"]

print(f"Episode {target_episode_index} 第一帧位姿:")
print(first_frame_pose)