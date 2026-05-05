import tf
import sys
import rospy
import numpy as np
import pandas as pd
sys.path.append("/home/k202/lerobot/src")
sys.path.append("/home/k202/lerobot")
from UR10e_deploy.robot_control import RobotOperation
from UR10e_deploy.transform_utils import convert_pose_quat2mat, convert_pose_euler2quat, convert_pose_euler2mat, \
    convert_pose_mat2quat


# df = pd.read_parquet('/home/k202/Insert_Notac_0504_ur10/Insert_plug/data/chunk-000/file-000.parquet', engine='pyarrow')
df = pd.read_parquet('/home/k202/Insert_Notac_0505_ur10_/Insert_plug/data/chunk-000/file-000.parquet', engine='pyarrow')
# 统计总轨迹数 (Episode 数量)
total_episodes = df['episode_index'].nunique()
print(f"实际存储的总轨迹数: {total_episodes}")

grouped_data = df.groupby('episode_index')

# 示例：获取第一个 Episode 的 state 和 action
first_ep_index = 20
first_ep_states = grouped_data.get_group(first_ep_index)['observation.state'].to_numpy()
first_ep_actions = grouped_data.get_group(first_ep_index)['action'].to_numpy()
print(f"Episode {first_ep_index} 的 State 数据维度: {first_ep_states.shape}")
print(f"Episode {first_ep_index} 的 Action 数据维度: {first_ep_actions.shape}")

rospy.init_node("UR10_Robot_Gripper_Publisher")
tf_listener = tf.TransformListener()
rospy.sleep(1)
(trans, rot) = tf_listener.lookupTransform('/tool0_controller', '/tool0', rospy.Time(0))
Ttool2tcp = np.array([trans[0], trans[1], trans[2], rot[0], rot[1], rot[2], rot[3]])
Ttool2tcp = convert_pose_quat2mat(Ttool2tcp)
robotoperation = RobotOperation(Ttool2tcp)

check_type = "state"
robotoperation.UR10_moveto_pose([[-0.4499,  0.6837,  0.1704,  0.9287, -0.0069,  0.0012, -0.3708]])
if check_type == "state":
    for i in range(first_ep_states.shape[0]):
        robotoperation.UR10_moveto_pose([first_ep_states[i][:7]])
        print(first_ep_states[i][:7])


# import pdb; pdb.set_trace()
# state1 = convert_pose_quat2mat(first_ep_states[80][:7]) # T_base_g1
# state2 = convert_pose_quat2mat(first_ep_states[81][:7]) # T_base_g2
# T_gt = np.linalg.inv(state1) @ state2   # T_g1_g2
# T_cal_state2 = state1 @ T_gt

# T_pred = convert_pose_quat2mat(first_ep_actions[80][:7])


if check_type == "action":
    # first_ep_actions = np.loadtxt("/home/k202/action_all.txt")[:, :7]
    robotoperation.UR10_moveto_pose([first_ep_states[0][:7]])
    start_pose = convert_pose_quat2mat(first_ep_states[0][:7])
    for i in range(first_ep_actions.shape[0]):
        # 实现方式1
        current_pose = robotoperation.get_ee_pose_moveit(return_quat = False)  # [4 4] T_base_current
        T_current_next = convert_pose_quat2mat(first_ep_actions[i][:7])
        T_base_next = current_pose @ T_current_next
        robotoperation.UR10_moveto_pose([convert_pose_mat2quat(T_base_next)])

        # # 实现方式2
        # T_gtgt = convert_pose_quat2mat(first_ep_actions[i][:7])
        # start_pose = start_pose @ T_gtgt
        # robotoperation.UR10_moveto_pose([convert_pose_mat2quat(start_pose)])

        # print("==================可靠位姿=======================")
        # print(robotoperation.get_ee_pose_moveit(return_quat = True))
        # print("==================机械臂读取不可靠位姿=======================")
        # print(convert_pose_mat2quat(current_pose))

        # print(convert_pose_mat2quat(T_base_next))
        # robotoperation.UR10_moveto_pose([convert_pose_mat2quat(T_base_next)])



# robotoperation.UR10_moveto_pose([[-0.4499,  0.6837,  0.1704,  0.9287, -0.0069,  0.0012, -0.3708]])















