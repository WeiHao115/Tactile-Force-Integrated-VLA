import matplotlib.pyplot as plt
import re
import numpy as np
import pandas as pd


def plot_loss_curve(log_file_path):
    steps = []
    losses = []

    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 使用正则表达式匹配 step 和 loss
            loss_match = re.search(r'loss:([0-9.]+)', line)
            if loss_match:
                loss = float(loss_match.group(1))
                losses.append(loss)

    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(losses)), losses, label='Loss', alpha=0.7, linewidth=1)
    plt.title('Training Loss Curve')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_and_save_comparison(log_file_path, save_path):
    """
    绘制并保存滤波前后六维力的对比图像。
    """
    steps = []
    losses = []
    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 使用正则表达式匹配 step 和 loss
            loss_match = re.search(r'loss:([0-9.]+)', line)
            if loss_match:
                loss = float(loss_match.group(1))
                losses.append(loss)
    losses = np.array(losses)
    print(losses.shape)
    timestamps = np.array(range(0, losses.shape[0]))
    original_data = losses


    def apply_edge_preserving_filter(force_data, median_window=31, mean_window=9):
        if force_data is None or len(force_data) == 0:
            return force_data
        # 将 numpy 数组转换为 DataFrame 以利用高效的 rolling 算子
        df = pd.DataFrame(force_data)
        df_med = df.rolling(window=median_window, center=True, min_periods=1).median()
        df_smooth = df_med.rolling(window=mean_window, center=True, min_periods=1).mean()
        return df_smooth.values.astype(np.float32)
    filtered_data = apply_edge_preserving_filter(losses)


    ft_cols = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']
    y_units = ['N', 'N', 'N', 'Nm', 'Nm', 'Nm']

    
    # 计算执行进度百分比 (%)
    progress = ((timestamps - timestamps[0]) / (timestamps[-1] - timestamps[0])) * 100
    fig, axes = plt.subplots(1, 1, figsize=(12, 18), sharex=True)
    axis_name = ft_cols
    axes.plot(progress, original_data, label=f'Original {axis_name}', color='blue', alpha=0.5, linewidth=1.2)
    axes.plot(progress, filtered_data, label=f'Filtered {axis_name}', color='red', alpha=0.8, linewidth=1.5)
    axes.set_ylabel(f'{axis_name} ({y_units})')
    axes.legend(loc='upper right')
    axes.grid(True, linestyle='--', alpha=0.6)
    axes.set_xlabel('Execution Progress (%)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()



if __name__ == "__main__":
    # plot_loss_curve('/home/8TDisk/0527model_decoder/metrics_log.txt')
    plot_and_save_comparison("/home/8TDisk/0604model_decoder_freeze_visionproj/metrics_log.txt",
                             "/home/k202/lerobot/test_dk/loss.png")




