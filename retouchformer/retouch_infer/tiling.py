# retouch_infer/tiling.py
import torch
import torch.nn.functional as F

def pad_to_multiple(x: torch.Tensor, multiple: int = 8):
    _, _, h, w = x.shape
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    return x, ph, pw

def unpad(x: torch.Tensor, ph: int, pw: int):
    if ph or pw:
        return x[..., : -ph if ph else None, : -pw if pw else None]
    return x
