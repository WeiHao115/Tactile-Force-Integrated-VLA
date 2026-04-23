import numpy as np

a1 = np.load("/home/k202/lerobot/UR10e_deploy/test_before_pre.npy")
a2 = np.load("/home/k202/lerobot/UR10e_deploy/test_after_pre.npy")


before_pre = np.load("/home/k202/lerobot/test_dk/before_pre.npz", allow_pickle=True)
after_pre = np.load("/home/k202/lerobot/test_dk/after_pre.npz", allow_pickle=True)

before_dict = before_pre['arr_0'].item()
after_dict = after_pre['arr_0'].item()
print("======================================")
print((before_dict["observation.state"] == after_dict["observation.state"]).all())
print(before_dict.keys())
print(after_dict.keys())
import pdb; pdb.set_trace()








