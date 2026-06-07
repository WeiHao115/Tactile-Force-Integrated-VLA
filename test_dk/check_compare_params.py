import torch
from safetensors.torch import load_file

# state_dict_14G = load_file("/home/ywl/pi05_lerobot/model.safetensors", device="cuda")
# print(len(state_dict_14G.keys()))
# print(state_dict_14G['paligemma_with_expert.gemma_expert.model.layers.15.mlp.down_proj.weight'])
# print(state_dict_14G['paligemma_with_expert.gemma_expert.model.layers.15.mlp.down_proj.weight'].dtype)

# state_dict_7G = load_file("/home/ywl/pi0_libero_finetuned/model.safetensors", device="cuda")
# print(len(state_dict_7G.keys()))
# print(state_dict_7G['model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.25.self_attn.v_proj.weight'])
# print(state_dict_7G['model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.25.self_attn.v_proj.weight'].dtype)


# with torch.no_grad():
#     state_dict_origin = load_file("/home/k202/pi05_lerobot/model.safetensors", device="cpu")
#     state_dict_5k = load_file("/home/k202/111/checkpoints/005000/pretrained_model/model.safetensors", device="cpu")
#     state_dict_10k = load_file("/home/k202/111/checkpoints/010000/pretrained_model/model.safetensors", device="cpu")
#     state_dict_15k = load_file("/home/k202/111/checkpoints/015000/pretrained_model/model.safetensors", device="cpu")


with torch.no_grad():
    origin_state_dict = load_file("/home/k202/pi05_lerobot/model.safetensors", device="cpu")
    trained_state_dict = load_file("/home/8TDisk/0528model_decoder_onlyforce/checkpoints/002000/pretrained_model/model.safetensors", device="cpu")

for name, value in origin_state_dict.items():
    # print(name, value.dtype)
    # print(name, trained_state_dict[name].dtype)
    print(value)
    print(trained_state_dict["model." + name])
    exit()

    if not (value == trained_state_dict["model." + name]).all():
        print(name)









# import torch

# def verify_tf32_computation():
#     if not torch.cuda.is_available():
#         print("未检测到 CUDA 环境。")
#         return

#     device_name = torch.cuda.get_device_name(0)
#     # 获取显卡的计算能力 (Compute Capability)
#     # Turing (TITAN) 是 7.5, Ampere (A40) 是 8.6
#     capability = torch.cuda.get_device_capability(0) 
    
#     print("-" * 50)
#     print(f"当前显卡型号: {device_name}")
#     print(f"硬件计算能力: {capability[0]}.{capability[1]}")
#     print("-" * 50)

#     # 构建两个较大的随机矩阵，以确保误差能够充分累加
#     N = 4096
#     torch.manual_seed(42)  # 固定随机种子确保矩阵一致
#     # 强制在显存中生成标准的 FP32 张量
#     A = torch.rand(N, N, dtype=torch.float32, device='cuda')
#     B = torch.rand(N, N, dtype=torch.float32, device='cuda')

#     # 测试 1：强制使用纯 FP32 计算
#     torch.backends.cuda.matmul.allow_tf32 = False
#     C_fp32 = torch.matmul(A, B)

#     # 测试 2：允许使用 TF32 计算
#     torch.backends.cuda.matmul.allow_tf32 = True
#     C_tf32 = torch.matmul(A, B)

#     # 计算两次矩阵乘法结果的最大绝对误差
#     max_diff = torch.abs(C_fp32 - C_tf32).max().item()
#     mean_diff = torch.abs(C_fp32 - C_tf32).mean().item()

#     print(f"开启与关闭 allow_tf32 的最大数值差异: {max_diff:.6f}")
#     print(f"开启与关闭 allow_tf32 的平均数值差异: {mean_diff:.6f}")
#     print("-" * 50)

#     # 逻辑判断
#     if max_diff > 1e-4:
#         print("结论: 计算产生了显著误差。")
#         print("说明底层确实执行了 TF32 尾数截断计算。你当前使用的是 TF32。")
#     else:
#         print("结论: 计算结果完全一致（误差为0）。")
#         print("说明代码指令 allow_tf32=True 被硬件忽略，底层始终在执行严格的 FP32 计算。")

# if __name__ == "__main__":
#     verify_tf32_computation()






