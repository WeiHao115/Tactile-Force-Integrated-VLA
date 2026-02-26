import torch
from safetensors.torch import load_file

state_dict_14G = load_file("/home/ywl/pi05_lerobot/model.safetensors", device="cuda")
print(len(state_dict_14G.keys()))
print(state_dict_14G['paligemma_with_expert.gemma_expert.model.layers.15.mlp.down_proj.weight'])
print(state_dict_14G['paligemma_with_expert.gemma_expert.model.layers.15.mlp.down_proj.weight'].dtype)

state_dict_7G = load_file("/home/ywl/pi0_libero_finetuned/model.safetensors", device="cuda")
# print(len(state_dict_7G.keys()))
# print(state_dict_7G['model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.25.self_attn.v_proj.weight'])
# print(state_dict_7G['model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.25.self_attn.v_proj.weight'].dtype)




