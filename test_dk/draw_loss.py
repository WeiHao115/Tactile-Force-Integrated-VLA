import matplotlib.pyplot as plt
import re
import numpy as np

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

if __name__ == "__main__":
    plot_loss_curve('/home/k202/111/metrics_log.txt')