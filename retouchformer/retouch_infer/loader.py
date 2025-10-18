# retouch_infer/loader.py
import importlib
import torch

def load_model(model_name: str,
               ckpt_dir: str = "release_model",
               epoch: str = "best",
               device: str = None):
    """
    加载 RetouchFormer 推理模型（InpaintGenerator）及权重
    """
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1) 动态导入并实例化
    net_mod = importlib.import_module('model.' + model_name)
    model = net_mod.InpaintGenerator().to(device)

    # 2) 读取权重（img_retouching.py 显示是“直接就是 state_dict”）
    ckpt_path = f"{ckpt_dir}/gen_{epoch}.pth"
    state = torch.load(ckpt_path, map_location=device)

    # 有些权重可能带 module. 前缀，统一处理一下（安全起见）
    new_state = {}
    for k, v in state.items():
        new_state[k.replace("module.", "")] = v

    model.load_state_dict(new_state, strict=False)
    model.eval()
    return model, device
