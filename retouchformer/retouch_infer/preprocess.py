import numpy as np
from PIL import Image
import torch

def read_image_to_tensor(path: str, resize_to: int = 512, debug: bool = False):
    """
    读取图片 -> resize到 (resize_to, resize_to) -> Tensor[1,C,H,W]，[-1,1]，RGB
    """
    img = Image.open(path).convert("RGB")
    if resize_to is not None:
        img = img.resize((resize_to, resize_to), Image.BICUBIC)

    arr = np.asarray(img).astype(np.float32) / 255.0       # [0,1]
    arr = (arr - 0.5) / 0.5                                # [-1,1]
    arr = np.transpose(arr, (2, 0, 1))                     # C,H,W
    t = torch.from_numpy(arr).unsqueeze(0)                 # 1,C,H,W

    if debug:
        print(f"[preprocess] tensor shape={tuple(t.shape)}, "
              f"range=({t.min().item():.3f},{t.max().item():.3f})")
    return t, img.size  # (W,H)
