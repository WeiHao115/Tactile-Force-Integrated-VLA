import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
import random

def solve_calibration_with_ransac(P_cap_list, T_rot_falan_list, 
                                 max_iterations=2000, 
                                 threshold=0.002, 
                                 min_inliers=8):
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
                    np.array([149.53869943856873,1094.9560438508408,439.0973709729452]),# 第1次测量
                    np.array([216.74441085828164,655.6598675031541,409.1247950562871]),
                    np.array([162.67211355863375,894.9592194087684,517.7071129162389]),
                    np.array([60.70289140109719,876.4163442214692,770.4394210818821]),
                    np.array([38.26624225425865,1288.8404814609676,430.85092015150815]),
                    np.array([328.35912443997177,273.71257371039235,373.05718232290803]),
                    np.array([18.38376661034617,365.77663605768515,294.5420382993048]),
                    np.array([583.6894966906859,697.4699800702477,354.6322342562836]),
                    np.array([8.303961812919772,859.6731202637257,330.71516750759105]),
                    np.array([-22.577447660329568,1085.3380082623196,465.39776326419195]),
                    np.array([-47.49230022396268,947.8535345184185,593.150731963107]),
                    np.array([35.215349140149435,702.6039523723554,352.84581425181193]),
                    np.array([333.1088473216591,892.6571527538271,449.7977185339173]),
                    np.array([148.3703341084631,550.7726303791674,188.1177628226323]),
                    np.array([464.6875315743403,238.91782783755883,492.6791166827589]),
                    np.array([249.86434786193868,1115.9144945459848,790.9762604571174]),
                    np.array([92.80788569121391,781.3405452713857,541.6274254673388]),
                    np.array([623.6431753848186,913.542336204435,439.03821912874423]),
                    np.array([-176.29024913960893,817.5038359998981,628.6955467321703]),
                    np.array([-318.26539414727034,902.3278514331242,721.249677955445]),
                    np.array([-261.61804628517154,805.5890072676226,310.5983479766823]),
                    np.array([48.232135877968176,744.9150692970296,338.70087655526197]),
                    np.array([109.26180340208867,613.9030555062288,353.2030569871356])

        # ... 所有的动捕数据写在这里
    ]) / 1000

    # 2. 填入机器人反馈的齐次矩阵 T_rot_falan (4x4)
    # 每一个矩阵对应上面同一个序号的动捕点
import numpy as np


my_T_rot_falan_list = [
    np.array([[ 0.995433,  0.010804, -0.094848, -0.401978],
        [ 0.074296, -0.711557,  0.69869 ,  0.344529],
        [-0.059941, -0.702546, -0.70911,   0.622078],
        [ 0.     ,   0.   ,     0.   ,     1.      ]]),

    np.array([[ 0.80563,  -0.453684,  0.38096,  -0.089083],
            [-0.589534, -0.550581 , 0.591025,  0.45597 ],
            [-0.05839 , -0.700736, -0.711027,  0.596895],
            [ 0. ,       0.    ,    0.   ,     1.      ]]),

    np.array([[ 0.784324, -0.506051, -0.358815, -0.138805],
            [-0.402398, -0.855232,  0.326579,  0.458534],
            [-0.472136, -0.111757, -0.874413 , 0.748723],
            [ 0. ,       0. ,       0.  ,      1.      ]]),

    np.array([[ 0.794612, -0.551607, -0.253617, -0.144542],
            [-0.258453, -0.685338,  0.680818,  0.266429],
            [-0.549358, -0.475438, -0.687143,  0.951828],
            [ 0.      ,  0.      ,  0.      ,  1.      ]]),
    np.array([[ 0.836978, -0.385404, -0.388499, -0.518984],
            [-0.464658, -0.875519, -0.132509,  0.434823],
            [-0.289069,  0.291426, -0.911872,  0.668922],
            [ 0.      ,  0.      ,  0.      ,  1.      ]]),
    np.array([[ 0.978503,  0.1262  , -0.163114,  0.430224],
            [ 0.187511, -0.873692,  0.44889 ,  0.619421],
            [-0.085862, -0.469825, -0.878574,  0.6102  ],
            [ 0.      ,  0.      ,  0.      ,  1.      ]]),
    np.array([[ 0.973769, -0.221539,  0.051907,  0.295923],
            [-0.218654, -0.847949,  0.48288 ,  0.297609],
            [-0.062963, -0.481564, -0.874147,  0.53024 ],
            [ 0.      ,  0.      ,  0.      ,  1.      ]]),
    np.array([[ 0.897085, -0.372177, -0.238167,  0.011336],
            [-0.282829, -0.897782,  0.337632,  0.884005],
            [-0.339481, -0.235524, -0.910649,  0.596081],
            [ 0.      ,  0.      ,  0.      ,  1.      ]]),
    np.array([[ 0.918123, -0.350325, -0.185265, -0.139194],
            [-0.193843, -0.804742,  0.561084,  0.247555],
            [-0.345652, -0.479232, -0.80676 ,  0.543969],
            [ 0.      ,  0.      ,  0.      ,  1.      ]]),
    np.array([[ 0.879112,  0.137713, -0.456286, -0.292666],
            [ 0.328012, -0.869378,  0.36958 ,  0.255483],
            [-0.345789, -0.474569, -0.809453,  0.676103],
            [ 0.      ,  0.      ,  0.      ,  1.      ]]),
    np.array([[ 0.937597, -0.152997, -0.312255, -0.19303 ],
            [ 0.132543, -0.672954,  0.727713,  0.145763],
            [-0.321471, -0.723688, -0.610681,  0.753776],
            [ 0.      ,  0.      ,  0.      ,  1.      ]]),
    np.array([[ 0.974703, -0.167002, -0.148538,  0.009   ],
            [-0.111919, -0.939966,  0.322393,  0.34035 ],
            [-0.193461, -0.297613, -0.934879,  0.601   ],
            [ 0.      ,  0.      ,  0.      ,  1.      ]]),
    np.array([[ 0.692904, -0.334466, -0.638762, -0.072321],
            [-0.064656, -0.911155,  0.406959,  0.60846 ],
            [-0.718125, -0.240684, -0.652968,  0.624839],
            [ 0.      ,  0.      ,  0.      ,  1.      ]]),
    np.array([[ 0.853336, -0.422214,  0.305864,  0.038953],
            [-0.441866, -0.897064, -0.005534,  0.544077],
            [ 0.276716, -0.130428, -0.952059,  0.442202],
            [ 0.      ,  0.      ,  0.      ,  1.      ]]),
    np.array([[ 0.712527, -0.516444,  0.474964,  0.290182],
            [-0.672822, -0.694936,  0.253721,  0.80306 ],
            [ 0.199036, -0.500349, -0.842636,  0.718606],
            [ 0.      ,  0.      ,  0.      ,  1.      ]]),
    np.array([[ 0.962626,  0.243252, -0.119076, -0.424939],
            [ 0.247601, -0.61227 ,  0.750879,  0.429937],
            [ 0.109747, -0.752299, -0.649616,  0.958412],
            [ 0.      ,  0.      ,  0.      ,  1.      ]]),
    np.array([[ 0.972726, -0.03598 ,  0.229149, -0.170381],
            [-0.175401, -0.76054 ,  0.625151,  0.318821],
            [ 0.151784, -0.648294, -0.746108,  0.737606],
            [ 0.      ,  0.      ,  0.      ,  1.      ]]),
    np.array([[ 0.998938, -0.024426, -0.039073, -0.257131],
            [-0.007576, -0.923474,  0.383586,  0.902753],
            [-0.045452, -0.382882, -0.922678,  0.680342],
            [ 0.      ,  0.      ,  0.      ,  1.      ]]),
    np.array([[ 0.967856, -0.251093,  0.014377, -0.141076],
            [-0.251219, -0.962454,  0.102813,  0.178092],
            [-0.011978, -0.10312 , -0.994597,  0.890764],
            [ 0.      ,  0.      ,  0.      ,  1.      ]]),
    np.array([[ 0.867781, -0.491755, -0.071647, -0.199168],
            [-0.222391, -0.51322 ,  0.828943, -0.150043],
            [-0.444408, -0.703408, -0.554725,  0.868321],
            [ 0.      ,  0.      ,  0.      ,  1.      ]]),
    np.array([[ 0.877234, -0.464793,  0.120121, -0.151969],
            [-0.479035, -0.831131,  0.282395,  0.051085],
            [-0.031419, -0.305268, -0.951748,  0.561696],
            [ 0.      ,  0.      ,  0.      ,  1.      ]]),
    np.array([[ 0.862144, -0.454592,  0.223727, -0.131699],
            [-0.505321, -0.739373,  0.444946,  0.322562],
            [-0.036851, -0.496662, -0.867162,  0.567229],
            [ 0.      ,  0.      ,  0.      ,  1.      ]]),
    np.array([[ 0.919071, -0.179915,  0.350627, -0.034302],
            [-0.345628, -0.795444,  0.497806,  0.37502 ],
            [ 0.189342, -0.578705, -0.793253,  0.563701],
            [ 0.      ,  0.      ,  0.      ,  1.      ]])

]
        # 所有的机器人位姿矩阵写在这里


    # 3. 调用函数进行计算
    # 这一行就是把上面准备好的数据“输入”进去
T_cap_rot, t_falan_marker, cost = solve_calibration_with_ransac(my_P_cap_list, my_T_rot_falan_list)
        
print("\n--- 求解结果 ---")
print(f"从 虚拟坐标系 到 base 的齐次变换矩阵 T:{np.linalg.inv(T_cap_rot)}")
print(f"从 tool0 到 marker点 的齐次变换矩阵 T{t_falan_marker}")
print(f"{cost}")