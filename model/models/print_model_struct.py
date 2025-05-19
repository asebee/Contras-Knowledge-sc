# import torch
# import pprint

# def show_model_structure(ckpt_path):
#     # 加载模型
#     ckpt = torch.load(ckpt_path, map_location='cpu')
    
#     # 打印模型的主要键
#     print("Checkpoint keys:")
#     print(list(ckpt.keys()))
    
#     # 如果包含 "model_state_dict" 或 "state_dict"
#     state_dict_key = None
#     if "model_state_dict" in ckpt:
#         state_dict_key = "model_state_dict"
#     elif "state_dict" in ckpt:
#         state_dict_key = "state_dict"
    
#     if state_dict_key:
#         state_dict = ckpt[state_dict_key]
        
#         # 打印层的名称
#         print("\nModel layers:")
#         for key in state_dict.keys():
#             print(f"- {key}, shape: {state_dict[key].shape}")
        
#         # 打印模型参数总量
#         total_params = sum(p.numel() for p in state_dict.values())
#         print(f"\nTotal parameters: {total_params:,}")
    
#     # 如果有其他元数据，打印它们
#     for key in ckpt.keys():
#         if key != state_dict_key and not isinstance(ckpt[key], torch.Tensor):
#             print(f"\n{key}:")
#             try:
#                 pprint.pprint(ckpt[key])
#             except:
#                 print("Unable to display this information")


# show_model_structure('models.ckpt')

import torch

# 加载模型
ckpt = torch.load('models.ckpt', map_location='cpu')

# 如果是标准的保存格式，查找 state_dict
state_dict = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))

# 查找输出层的权重矩阵
# 通常是最后的线性层或投影层，例如可能命名为:
output_layer_candidates = [k for k in state_dict.keys() if 'output' in k or 'out' in k or 'final' in k or 'head' in k]
for key in output_layer_candidates:
    if 'weight' in key:
        print(f"可能的输出层: {key}, 维度: {state_dict[key].shape}")
        # 第一个维度通常是输出维度
        output_dim = state_dict[key].shape[0]
        print(f"输出维度可能是: {output_dim}")