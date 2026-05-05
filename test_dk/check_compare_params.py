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
    trained_state_dict = load_file("/home/8TDisk/model0503_umiv1/checkpoints/003000/pretrained_model/model.safetensors", device="cpu")
for name, value in trained_state_dict.items():
    print(name)
    # try:
    #     if not (value == trained_state_dict["model." + name]).all():
    #         print(name)
    # except:
    #     print("error")
    #     continue
















