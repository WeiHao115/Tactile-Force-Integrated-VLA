import numpy as np
import re
import os
import cv2

umi_file_path = '/home/k202/0416/yingshe/umi_body_abs.txt'
gripper_file_path = '/home/k202/0416/yingshe/gripper_state.txt'
output_file_path = '/home/k202/0416/yingshe/gripper_state_time.txt'
TAC_PATH = '/home/k202/0416/yingshe/tactile_left'
output_file_pathV2 = '/home/k202/0416/yingshe/gripper_state_timeV2.txt'
YUZHI = 70


def process_gripper_states(umi_file, gripper_file, output_file):
    """
    根据时间戳将夹爪状态匹配到机械臂位姿数据中。
    
    参数:
    umi_file: 机械臂位姿数据文件路径
    gripper_file: 夹爪状态日志文件路径
    output_file: 匹配结果的输出保存路径
    """
    
    # 1. 解析夹爪状态文件
    transitions = []
    with open(gripper_file, 'r', encoding='utf-8') as f:
        for line in f:
            # 使用正则表达式提取时间戳和状态
            # 匹配格式: [1774850895.424666] MCU_Time: 1626020ms | Status: 1
            match = re.search(r'\[(.*?)\].*Status:\s*(\d)', line)
            if match:
                ts = float(match.group(1))
                status = int(match.group(2))
                transitions.append((ts, status))
                
    if not transitions:
        raise ValueError("未在夹爪状态文件中找到有效的状态转换数据。")

    # 按照时间戳升序排序，确保时序正确
    transitions.sort(key=lambda x: x[0])
    trans_ts = np.array([x[0] for x in transitions])
    trans_status = np.array([x[1] for x in transitions])

    # 2. 读取机械臂位姿文件
    # 假设数据以空格或制表符分隔
    umi_data = np.loadtxt(umi_file)
    umi_ts = umi_data[:, 0]
    N = len(umi_ts)

    # 3. 匹配逻辑 (核心)
    # 使用二分查找获取 umi_ts 在 trans_ts 中的插入位置
    # side='right' 保证取到的是当前时间戳“左侧最近”的一次状态变化
    idx = np.searchsorted(trans_ts, umi_ts, side='right') - 1

    # 初始化状态数组
    # 假设在第一次状态发生改变之前的默认夹爪状态为 0
    matched_status = np.zeros(N, dtype=int)
    
    # 如果索引 >= 0，说明该时间戳在第一次状态变化之后，直接赋予对应的状态
    valid_mask = idx >= 0
    matched_status[valid_mask] = trans_status[idx[valid_mask]]

    # 4. 组装结果并保存为 [N, 2] 格式
    result = np.column_stack((umi_ts, matched_status))

    # 设置保存格式：时间戳保留 6 位小数，状态为整数
    np.savetxt(output_file, result, fmt='%.6f %d')


class Tac_Cen():
    def __init__(self):
        self.tactile_center = None
        self.time = 0
        self.min = 0
        self.tac_time = 0
        self.idx = 0
        self.diff_list = []
        self.gipper_time = []
        self.new_time = []

    def jiance(self):
        # 1. 遍历./gripper_state_time.txt
        # 2. 找到0-1突变的位置
        # 从突变的位置索引开始遍历，找出时间戳最近的shichu图像，把中心区域裁剪出来，付给self.tactile_center 
        # 从突变位置的第2帧开始，和self.tactile_center对比
        # 变化超过一个阈值，就认为达到了，gripper_state_time.txt中才开始从0变成1
        # 3. 开夹爪的时候，要去更新self.tactile_center
        with open(output_file_path, "r", encoding="utf-8") as f:
            for i,line in enumerate(f):
                if int(line.split()[1]):
                    self.time = float(line.split()[0])  
                    print(self.time)
                    break

        files = sorted(os.listdir(TAC_PATH))
        self.min = abs(self.time- float(files[0][:-4]))
        self.tac_time = float(files[0][:-4])
        self.tactile_center = cv2.imread(f'{TAC_PATH}/{files[0]}')[56:168,56:168]
        
        for i in range(len(files)):
            cur_time = abs(self.time- float(files[i][:-4]))
            if cur_time<self.min:
                self.min = cur_time
                self.tac_time = float(files[i][:-4])
                self.idx = i
        print(self.tac_time)

        for j in range(self.idx,len(files)):
            picture = cv2.imread(f'{TAC_PATH}/{files[j]}')
            diff = cv2.absdiff(picture[56:168,56:168],self.tactile_center)
            diff_float = float(np.mean(np.square(diff.astype(np.float32))))#计算均方误差
            if diff_float>YUZHI:
                self.gipper_time.append(float(files[j][:-4]))
            self.diff_list.append(diff_float)
        print(self.diff_list) 
        # print(self.gipper_time)

        with open(output_file_path,"r",encoding="utf-8") as f1:
            lines = f1.readlines()

        for line in lines:
            if float(line.split()[0])<self.gipper_time[0] or float(line.split()[0])>self.gipper_time[-1]:
                self.new_time.append(f"{float(line.split()[0])}\t0\n")
            else:
                self.new_time.append(f"{float(line.split()[0])}\t1\n")
        
        with open(output_file_pathV2, "w", encoding="utf-8") as f2:
            f2.writelines(self.new_time) 
          

if __name__ == "__main__":
    
    process_gripper_states(umi_file_path, gripper_file_path, output_file_path)

    tac_cen = Tac_Cen()
    tac_cen.jiance()