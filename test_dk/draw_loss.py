# import matplotlib.pyplot as plt
# import re
# import numpy as np

# def plot_loss_curve(log_file_path):
#     steps = []
#     losses = []

#     with open(log_file_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             # 使用正则表达式匹配 step 和 loss
#             loss_match = re.search(r'loss:([0-9.]+)', line)
#             if loss_match:
#                 loss = float(loss_match.group(1))
#                 losses.append(loss)

#     plt.figure(figsize=(10, 6))
#     plt.plot(np.arange(len(losses)), losses, label='Loss', alpha=0.7, linewidth=1)
#     plt.title('Training Loss Curve')
#     plt.xlabel('Step')
#     plt.ylabel('Loss')
#     plt.grid(True, linestyle='--', alpha=0.6)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     plot_loss_curve('/home/8TDisk/model0502_ur10v2/metrics_log.txt')









import pandas as pd

# 读取 Parquet 文件
df = pd.read_parquet('/home/k202/Insert_Notac_0503_ur10/Insert_plug/meta/episodes/chunk-000/file-000.parquet', engine='pyarrow')

# 统计总轨迹数 (Episode 数量)
total_episodes = df['episode_index'].nunique()
print(f"实际存储的总轨迹数: {total_episodes}")

# 查看轨迹索引的最大值和最小值
min_ep = df['episode_index'].min()
max_ep = df['episode_index'].max()
print(f"轨迹索引范围: 从 {min_ep} 到 {max_ep}")



