import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
import random

def solve_calibration_with_ransac(P_cap_list, T_rot_falan_list, 
                                 max_iterations=100, 
                                 threshold=0.002, 
                                 min_inliers=10):
    """
    使用 RANSAC 剔除外点后的高精度标定求解
    threshold: 判定为内点的残差阈值（单位：米，根据动捕精度调整，如 5mm）
    """
    
    P_cap_list = np.array(P_cap_list)
    T_rot_falan_list = np.array(T_rot_falan_list)
    num_samples = len(P_cap_list)
    
    best_inliers_mask = None
    best_error = np.inf
    final_params = None

    def residuals(params, P_cap, T_rf):
        r_vec = params[0:3]
        t_vec = params[3:6]
        y_pos = params[6:9]
        
        rot_cap_res = R.from_rotvec(r_vec).as_matrix()
        
        preds = []
        for i in range(len(P_cap)):
            R_rf = T_rf[i][:3, :3]
            t_rf = T_rf[i][:3, 3]
            prediction = rot_cap_res @ (R_rf @ y_pos + t_rf) + t_vec
            preds.append(prediction)
        
        return np.array(preds) - P_cap

    # RANSAC 主循环
    for i in range(max_iterations):
        # 1. 随机选择最小样本集 (至少需要 4 组点以保证解的稳定性)
        indices = random.sample(range(num_samples), k=max(4, int(num_samples * 0.2)))
        sample_P = P_cap_list[indices]
        sample_T = T_rot_falan_list[indices]
        
        # 2. 初始估计与局部优化
        initial_guess = np.zeros(9)
        res_sample = least_squares(
            lambda p, x, y: residuals(p, x, y).flatten(),
            initial_guess,
            args=(sample_P, sample_T),
            method='lm'
        )
        
        # 3. 计算所有点的残差，筛选内点
        all_res = residuals(res_sample.x, P_cap_list, T_rot_falan_list)
        dist = np.linalg.norm(all_res, axis=1)
        inliers_mask = dist < threshold
        num_inliers = np.sum(inliers_mask)
        
        # 4. 如果内点足够多，记录当前最佳模型
        if num_inliers >= min_inliers:
            current_error = np.mean(dist[inliers_mask])
            if num_inliers > (0 if best_inliers_mask is None else np.sum(best_inliers_mask)):
                best_inliers_mask = inliers_mask
                best_error = current_error
                final_params = res_sample.x

    if best_inliers_mask is None:
        raise ValueError("RANSAC 未能找到足够的内点，请检查阈值或数据质量。")

    # 5. 使用所有内点进行最终的全局非线性优化（Refine）
    print(f"RANSAC 完成。内点数量: {np.sum(best_inliers_mask)}/{num_samples}")
    
    final_res = least_squares(
        lambda p, x, y: residuals(p, x, y).flatten(),
        final_params,
        args=(P_cap_list[best_inliers_mask], T_rot_falan_list[best_inliers_mask]),
        method='trf',
        loss='soft_l1', # 二次加固：即使在内点中也处理微小噪声
        xtol=1e-12,
        ftol=1e-12,
        verbose=1
    )

    # 解析最终结果
    best_p = final_res.x
    T_cap_rot = np.eye(4)
    T_cap_rot[:3, :3] = R.from_rotvec(best_p[0:3]).as_matrix()
    T_cap_rot[:3, 3] = best_p[3:6]
    res_t_falan_gripper = best_p[6:9]

    return T_cap_rot, res_t_falan_gripper, final_res.cost


def solve_calibration(P_cap_list, T_rot_falan_list):
    """
    P_cap_list: List of np.array([x, y, z]), 动捕测得的点坐标
    T_rot_falan_list: List of np.array([4, 4]), 机器人反馈的齐次矩阵
    """
    
    # 1. 定义残差函数
    def residuals(params, P_cap, T_rf):
        # 提取参数
        r_vec = params[0:3]  # cap_to_rot 的旋转轴角
        t_vec = params[3:6]  # cap_to_rot 的平移
        y_pos = params[6:9]  # falan_to_marker 的平移
        
        rot_cap_res = R.from_rotvec(r_vec).as_matrix()
        
        res = []
        for i in range(len(P_cap)):
            # 机器人末端当前的旋转和平移
            R_rf = T_rf[i][:3, :3]
            t_rf = T_rf[i][:3, 3]
            
            # 预测点在动捕系下的坐标: P = R_cr * (R_rf * y + t_rf) + t_cr
            # 注意：这里的逻辑是 P_cap 是目标点，我们建立从 marker 到 cap 的变换
            # 这里的公式根据你的 T_cap_rot 定义可能需要微调方向
            # 按照你的公式 P_cap = get_pingyi(T_cap_rot * T_rot_falan * T_falan_gripper)
            prediction = rot_cap_res @ (R_rf @ y_pos + t_rf) + t_vec
            res.append(prediction - P_cap[i])
            
        return np.array(res).flatten()

    # 2. 初始猜测 (可以使用简单的平均值或恒等变换)
    # params: [r1, r2, r3, t1, t2, t3, y1, y2, y3]
    initial_guess = np.zeros(9)
    
    print("开始高精度非线性优化...")
    
    # 3. 使用 least_squares 求解
    # 'lm' 适合高精度，'trf' 适合带边界约束
    result = least_squares(
        residuals, 
        initial_guess, 
        args=(P_cap_list, T_rot_falan_list),
        method='trf', 
        loss='soft_l1', # 鲁棒核函数，防止个别动捕跳点干扰
        xtol=1e-12,     # 极高的收敛精度
        ftol=1e-12,
        verbose=2
    )
    
    # 4. 解析结果
    best_params = result.x
    res_R_cap_rot = R.from_rotvec(best_params[0:3]).as_matrix()
    res_t_cap_rot = best_params[3:6]
    res_t_falan_gripper = best_params[6:9]
    
    # 组装 T_cap_rot
    T_cap_rot = np.eye(4)
    T_cap_rot[:3, :3] = res_R_cap_rot
    T_cap_rot[:3, 3] = res_t_cap_rot
    
    return T_cap_rot, res_t_falan_gripper, result.cost

# --- 模拟测试 ---
if __name__ == "__main__":
    # --- 你的实际数据输入区 ---

    # 1. 填入动捕测得的坐标 P_cap (单位：mm)
    # 每一行代表一个采样点 [x, y, z]
    my_P_cap_list = np.array([
        np.array([266.2657438397335,973.0444069300376,548.2481046836322]), # 第1次测量
                  np.array([323.84367733081496,538.8204505244653,535.8443010535149]),
                    np.array([165.8758997617536, 777.375509816562, 398.1093794864003]),
                    np.array([457.9409040024498, 747.1995874492925, 510.46470008195746]),
                    np.array([447.46474503237306, 1024.2109471449833, 374.6346124813954]),
                    np.array([557.2595762778333, 514.7606847962119, 553.8805112204436]),
                    np.array([136.06449732519536, 661.2750705586834, 335.3140989020275]),
                    np.array([554.7668132560049, 371.48264664927024, 435.20803493483385]),
                    np.array([755.5135772014332, 724.9940454779165, 319.1445299666497]),
                    np.array([548.1393130721262, 1055.1274409677712, 430.643422454325]),
                    np.array([330.45934864509115, 845.6999726205283, 255.847544829456]),
                    np.array([405.32810357066205, 513.7468318553149, 621.9078190555815]),
                    np.array([128.69158194502106, 631.3725206473663, 119.79842572862654]),
                    np.array([8.15919649913418, 536.2953936580931, 223.9105175682492]),
                    np.array([-69.7007495658881, 1013.2628834552489, 193.57536692012025]),
                    np.array([98.14440952473107, 1017.9733709482182, 550.3598051603615]),
                    np.array([443.6257622844188, 721.6021078269827, 585.4063640807195]),
                    np.array([467.516175636797, 336.8355613528851, 390.87287657337606]),
                    np.array([372.7107487742477, 548.4034316446014, 149.23411224153386]),
                    np.array([744.0404959296299, 762.9773402597867, 346.2250771196185]),
                    np.array([647.61099325739, 682.7705641927538, 564.0048465890102]),
 # 第3次测量
        # ... 把你所有的动捕数据写在这里
    ]) / 1000

    # 2. 填入机器人反馈的齐次矩阵 T_rot_falan (4x4)
    # 每一个矩阵对应上面同一个序号的动捕点
    my_T_rot_falan_list = [
np.array([
    [0.935277, 0.346084, -0.074047, -0.296337],
    [0.31835, -0.731254, 0.603259, 0.520783],
    [0.154631, -0.587787, -0.794101, 0.71289],
    [0., 0., 0., 1.]
]),
np.array([
    [0.949108, -0.139737, 0.282253, 0.059792],
    [-0.274976, -0.804614, 0.526292, 0.604481],
    [0.153562, -0.577121, -0.802091, 0.705284],
    [0., 0., 0., 1.]
]),
np.array([
    [0.928296, 0.364211, 0.074943, -0.127292],
    [0.279961, -0.817218, 0.503762, 0.444526],
    [0.24472, -0.44666, -0.860585, 0.578264],
    [0., 0., 0., 1.]
]),
np.array([
    [0.927574, 0.365865, 0.075825, -0.105149],
    [0.267229, -0.791437, 0.549742, 0.728118],
    [0.261142, -0.489664, -0.831886, 0.684976],
    [0., 0., 0., 1.]
]),
np.array([
    [0.786668, 0.613567, -0.068477, -0.349607],
    [0.583019, -0.701822, 0.40931, 0.738031],
    [0.203081, -0.361915, -0.909822, 0.562527],
    [0., 0., 0., 1.]
]),
np.array([
    [0.959066, 0.186146, 0.213407, 0.0941],
    [0.033259, -0.822426, 0.567899, 0.83078],
    [0.281224, -0.537555, -0.794952, 0.722446],
    [0., 0., 0., 1.]
]),
np.array([
    [0.971943, 0.161085, 0.171403, -0.031501],
    [0.075307, -0.903448, 0.422031, 0.434431],
    [0.222836, -0.397282, -0.890231, 0.522893],
    [0., 0., 0., 1.]
]),
np.array([
    [0.968638, 0.043854, 0.244574, 0.230828],
    [-0.070179, -0.895941, 0.438594, 0.856805],
    [0.238358, -0.442003, -0.864765, 0.619287],
    [0., 0., 0., 1.]
]),
np.array([
    [0.928304, 0.366562, 0.062326, -0.086995],
    [0.3016, -0.840363, 0.450363, 1.045853],
    [0.217463, -0.399276, -0.890667, 0.505099],
    [0., 0., 0., 1.]
]),
np.array([
    [0.78659, 0.610386, -0.093298, -0.378631],
    [0.570708, -0.660985, 0.487229, 0.821982],
    [0.235729, -0.436495, -0.868276, 0.608903],
    [0., 0., 0., 1.]
]),
np.array([
    [0.886457, 0.461584, 0.03368, -0.189365],
    [0.462097, -0.886784, -0.00901, 0.708829],
    [0.025708, 0.023551, -0.999392, 0.465886],
    [0., 0., 0., 1.]
]),
np.array([
    [0.927816, 0.194936, 0.318053, 0.077574],
    [-0.107022, -0.677662, 0.727544, 0.647241],
    [0.357357, -0.709066, -0.607883, 0.751388],
    [0., 0., 0., 1.]
]),
np.array([
    [0.986301, 0.084098, 0.141907, 0.00577],
    [0.043672, -0.962709, 0.26699, 0.459042],
    [0.159068, -0.257135, -0.953194, 0.321025],
    [0., 0., 0., 1.]
]),
np.array([
    [0.899487, -0.14067, 0.413684, 0.046048],
    [-0.333109, -0.833468, 0.440875, 0.305849],
    [0.282775, -0.534363, -0.796552, 0.392076],
    [0., 0., 0., 1.]
]),
np.array([
    [0.726277, 0.406766, -0.554133, -0.228138],
    [0.663668, -0.62493, 0.411105, 0.223818],
    [-0.179071, -0.666336, -0.72383, 0.342337],
    [0., 0., 0., 1.]
]),
np.array([
    [0.820855, 0.234936, -0.520578, -0.248226],
    [0.195557, -0.971997, -0.130303, 0.499798],
    [-0.536613, 0.005157, -0.843812, 0.729806],
    [0., 0., 0., 1.]
]),
np.array([
    [0.840134, -0.142328, -0.523372, 0.036069],
    [-0.102274, -0.989216, 0.104838, 0.804429],
    [-0.532649, -0.034551, -0.845631, 0.766686],
    [0., 0., 0., 1.]
]),
np.array([
    [0.88753, -0.40009, -0.228513, 0.359034],
    [-0.393617, -0.916185, 0.075311, 0.841937],
    [-0.239491, 0.023106, -0.970624, 0.599845],
    [0., 0., 0., 1.]
]),
np.array([
    [0.89184, -0.213569, -0.39876, 0.187984],
    [-0.345503, -0.890598, -0.29574, 0.81755],
    [-0.291974, 0.401525, -0.86806, 0.338556],
    [0., 0., 0., 1.]
]),
np.array([
    [0.969187, 0.090436, -0.229126, -0.069266],
    [0.10606, -0.992737, 0.056793, 1.112822],
    [-0.222325, -0.079344, -0.971739, 0.55142],
    [0., 0., 0., 1.]
]),
np.array([
    [0.995462, 0.043327, -0.084721, -0.018115],
    [0.062388, -0.969437, 0.237275, 0.982866],
    [-0.071851, -0.241484, -0.967741, 0.768369],
    [0., 0., 0., 1.]
])
        # ... 把你所有的机器人位姿矩阵写在这里
    ]

    # 3. 调用函数进行计算
    # 这一行就是把上面准备好的数据“输入”进去
    T_cap_rot, t_falan_marker, cost = solve_calibration_with_ransac(my_P_cap_list, my_T_rot_falan_list)
        
    print("\n--- 求解结果 ---")
    print(f"从 虚拟坐标系 到 base 的齐次变换矩阵 T:{np.linalg.inv(T_cap_rot)}")
    print(f"从 tool0 到 marker点 的齐次变换矩阵 T{t_falan_marker}")
    print(f"{cost}")



