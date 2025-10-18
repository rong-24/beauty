# retouch_infer/postprocess.py
import os
import numpy as np
from PIL import Image
import torch
from torchvision.utils import save_image

def tensor_to_image(t: torch.Tensor):
    """
    模型输出 pred_img 为 [-1,1]；转换到 uint8 RGB
    """
    t = t.detach().cpu().clamp(-1, 1)
    t = (t * 0.5 + 0.5)                                    # [0,1]
    t = (t * 255.0).round().byte()
    t = t.squeeze(0).permute(1, 2, 0).numpy()
    return Image.fromarray(t)

def save_tensor_normed(pred_img: torch.Tensor, path: str):
    """
    与原脚本等价保存：save_image(normalize=True, value_range=(-1,1))
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_image(pred_img, path, normalize=True, value_range=(-1, 1))
