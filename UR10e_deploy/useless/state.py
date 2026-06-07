from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 1. 加载数据集 (以官方示例为例)
repo_id = "/home/ywl/test319/test_data_lerobotdataset/tactile_manipulation_test"
dataset = LeRobotDataset(repo_id)

# 2. 获取特定帧的数据
print("第 0 帧")
frame_data = dataset[0]

# 3. 提取状态
# 在 LeRobot 标准格式中，状态通常存储在 'observation.state' 键下
state = frame_data["observation.state"]

print(f"State shape: {state.shape}")
print(f"State tensor: {state}")

print("最后1帧")
frame_data = dataset[-1]

state = frame_data["observation.state"]

print(f"State shape: {state.shape}")
print(f"State tensor: {state}")