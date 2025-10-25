# -*- coding: utf-8 -*-
import os, sys, os.path as osp
from typing import Optional, Union
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as T
from torch import nn

# 允许从当前目录作为包导入
HERE = os.path.abspath(os.path.dirname(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from CSD_MT.options import Options
from CSD_MT.model import CSD_MT
from faceutils.face_parsing.model import BiSeNet

ArrayLike = Union[np.ndarray, Image.Image]

class CSDMTService:
    def __init__(
        self,
        csdmt_weights: str = osp.join(HERE, "CSD_MT/weights/CSD_MT.pth"),
        faceparsing_weights: str = osp.join(HERE, "faceutils/face_parsing/res/cp/79999_iter.pth"),
        device: Optional[str] = None,
        align_mode: str = "skip",
        resize: int = 256,
    ):
        print("========== [CSDMTService.__init__] ==========")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.resize = resize
        self.align_mode = align_mode
        print(f"[Init] device        : {self.device}")
        print(f"[Init] resize        : {self.resize}")
        print(f"[Init] align_mode    : {self.align_mode}")
        print(f"[Init] CSD-MT weights: {csdmt_weights}")
        print(f"[Init] FP  weights   : {faceparsing_weights}")

        # Face parsing
        print("[Init] Loading BiSeNet (face parsing) ...")
        self.n_classes = 19
        self.face_parsing = BiSeNet(n_classes=self.n_classes)
        state = torch.load(faceparsing_weights, map_location="cpu")
        self.face_parsing.load_state_dict(state)
        self.face_parsing.to(self.device).eval()
        print("[Init] BiSeNet loaded and set to eval().")

        # CSD-MT
        print("[Init] Parsing CSD-MT options (detached from CLI) ...")
        parser = Options()
        _argv_backup = sys.argv[:]
        try:
            sys.argv = [sys.argv[0]]
            self.opts = parser.parse()
        finally:
            sys.argv = _argv_backup

        self.opts.resize_size = resize
        print(f"[Init] opts.resize_size => {self.opts.resize_size}")

        print("[Init] Building CSD-MT model ...")
        self.makeup_model = CSD_MT(self.opts)
        _ = self.makeup_model.resume(csdmt_weights)  # 会把 content_style_separation.kernel 从 ckpt 里加载进来
        self.makeup_model.to(self.device).eval()
        # 覆盖内部 self.gpu，让 test_pair() 不再把输入搬回 CPU
        try:
            self.makeup_model.gpu = torch.device(str(self.device))
            print(f"[Init] Override model.gpu -> {self.makeup_model.gpu}")
        except Exception as e:
            print("[Init] Cannot override model.gpu:", e)

        self.makeup_model.float()  # 统一 float32，避免 half/float 冲突

        try:
            # 某些老项目依赖 BaseModel.Tensor
            self.makeup_model.gpu_ids = [0] if self.device.type == "cuda" else []
            self.makeup_model.Tensor = torch.cuda.FloatTensor if self.device.type == "cuda" else torch.FloatTensor
        except Exception:
            pass

        try:
            p0 = next(self.makeup_model.parameters())
            print(f"[Init] CSD-MT param dtype={p0.dtype}, device={p0.device}")
        except StopIteration:
            print("[Init] CSD-MT model has no parameters?")

        # 兜底：把误入 CPU 的输入在每层前搬回 CUDA
        def _force_cuda_hook(module, args):
            if self.device.type != "cuda":
                return args
            new_args = []
            for a in args:
                if isinstance(a, torch.Tensor) and a.device.type != "cuda":
                    new_args.append(a.to(self.device, non_blocking=True))
                else:
                    new_args.append(a)
            return tuple(new_args)

        hook_count = 0
        for m in self.makeup_model.modules():
            if isinstance(m, (nn.Conv2d, nn.BatchNorm2d, nn.InstanceNorm2d, nn.ReLU, nn.LeakyReLU, nn.Sequential)):
                m.register_forward_pre_hook(_force_cuda_hook)
                hook_count += 1
        print(f"[Init] Registered {hook_count} forward_pre_hooks to force CUDA inputs")
        print("========== [CSDMTService.__init__ DONE] ==========\n")

        self._to_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    # ---------- alignment ----------
    def _maybe_align(self, image: ArrayLike) -> np.ndarray:
        if isinstance(image, Image.Image):
            img = image.convert("RGB")
        else:
            img = Image.fromarray(image).convert("RGB")

        if self.align_mode == "skip":
            return np.array(img)

        if self.align_mode == "dlib":
            print("[Align] Using dlib alignment ...")
            from faceutils import get_dlib
            futils = get_dlib()
            up_ratio   = 0.2 / 0.85
            down_ratio = 0.15 / 0.85
            width_ratio= 0.2 / 0.85
            faces = futils.detect(img)
            if not faces:
                raise ValueError("No face detected for dlib alignment.")
            image_aligned, _, _ = futils.crop(img, faces[0], up_ratio, down_ratio, width_ratio)
            return np.array(image_aligned)

        raise ValueError(f"Unknown align_mode: {self.align_mode}")

    # ---------- parsing ----------
    @torch.no_grad()
    def _face_parsing_19(self, x: np.ndarray) -> np.ndarray:
        img = Image.fromarray(x).resize((512, 512), Image.BILINEAR)
        ten = self._to_tensor(img).unsqueeze(0).to(self.device)
        out = self.face_parsing(ten)[0]
        parsing = out.squeeze(0).argmax(0).detach().to("cpu").numpy().astype(np.int32)
        return parsing

    # ---------- 19 -> 10 ----------
    def _split_parse(self, parse: np.ndarray) -> np.ndarray:
        h, w = parse.shape
        c = self.opts.semantic_dim
        result = np.zeros([h, w, c], dtype=np.float32)
        result[:, :, 0][np.isin(parse, [0, 16, 17, 18, 9])] = 1
        result[:, :, 1][np.isin(parse, [1, 6])] = 1
        result[:, :, 2][np.isin(parse, [2, 3])] = 1
        result[:, :, 3][np.isin(parse, [4, 5])] = 1
        result[:, :, 4][np.isin(parse, [7, 8])] = 1
        result[:, :, 5][parse == 10] = 1
        result[:, :, 6][parse == 11] = 1
        result[:, :, 7][parse == 12] = 1
        result[:, :, 8][parse == 13] = 1
        result[:, :, 9][np.isin(parse, [14, 15])] = 1
        return result

    # ---------- local masks ----------
    def _local_masks(self, split_parse: np.ndarray) -> np.ndarray:
        h, w, _ = split_parse.shape
        all_mask = np.zeros([h, w], dtype=np.float32)
        all_mask[split_parse[:, :, 0] == 0] = 1
        all_mask[split_parse[:, :, 3] == 1] = 0
        all_mask[split_parse[:, :, 6] == 1] = 0
        all_mask = np.repeat(all_mask[:, :, None], 3, axis=2)
        return all_mask

    # ---------- main ----------
    @torch.no_grad()
    def transfer(self, content: ArrayLike, style: ArrayLike) -> np.ndarray:
        print("========== [CSDMTService.transfer] ==========")
        print("[Stage] maybe_align(content) ...")
        c_img = self._maybe_align(content)
        print("[Info ] content after align:", c_img.shape, c_img.dtype)

        print("[Stage] maybe_align(style) ...")
        s_img = self._maybe_align(style)
        print("[Info ] style   after align:", s_img.shape, s_img.dtype)

        print(f"[Stage] resize to {self.resize}x{self.resize} ...")
        c_img = np.array(Image.fromarray(c_img).resize((self.resize, self.resize), Image.BILINEAR))
        s_img = np.array(Image.fromarray(s_img).resize((self.resize, self.resize), Image.BILINEAR))
        print("[Info ] content resized:", c_img.shape, c_img.dtype)
        print("[Info ] style   resized:", s_img.shape, s_img.dtype)

        print("[Stage] face parsing (content) ...")
        c_parse = self._face_parsing_19(c_img)
        print("[Info ] c_parse:", c_parse.shape, c_parse.dtype)

        print("[Stage] face parsing (style) ...")
        s_parse = self._face_parsing_19(s_img)
        print("[Info ] s_parse:", s_parse.shape, s_parse.dtype)

        print("[Stage] resize parses to model size (NEAREST) ...")
        c_parse = np.array(Image.fromarray(c_parse).resize((self.resize, self.resize), Image.NEAREST))
        s_parse = np.array(Image.fromarray(s_parse).resize((self.resize, self.resize), Image.NEAREST))

        print("[Stage] split_parse ...")
        c_split = self._split_parse(c_parse)
        s_split = self._split_parse(s_parse)
        print("[Info ] c_split:", c_split.shape, c_split.dtype)
        print("[Info ] s_split:", s_split.shape, s_split.dtype)

        print("[Stage] build local masks ...")
        c_all_m = self._local_masks(c_split)
        s_all_m = self._local_masks(s_split)
        print("[Info ] c_all_m:", c_all_m.shape, c_all_m.dtype)
        print("[Info ] s_all_m:", s_all_m.shape, s_all_m.dtype)

        print("[Stage] pack tensors ...")

        def prep_img(np_img):
            img = (np_img.astype(np.float32) / 127.5) - 1.0
            img = np.transpose(img, (2, 0, 1))           # HWC -> CHW
            ten = torch.from_numpy(img).unsqueeze(0)     # (1,3,H,W) CPU
            return ten.to(self.device, non_blocking=True)

        def prep_mask(np_mask):
            if np_mask.ndim == 3:  # HWC
                m = np.transpose(np_mask.astype(np.float32), (2, 0, 1))
            else:
                m = np_mask.astype(np.float32)
                if m.ndim == 2:
                    m = m[None]
            ten = torch.from_numpy(m).unsqueeze(0)       # (1,C,H,W) CPU
            return ten.to(self.device, non_blocking=True)

        c_img_t   = prep_img(c_img)
        s_img_t   = prep_img(s_img)
        c_split_t = prep_mask(c_split)
        s_split_t = prep_mask(s_split)
        c_all_m_t = prep_mask(c_all_m)
        s_all_m_t = prep_mask(s_all_m)

        print("[Info ] devices:",
              c_img_t.device, c_split_t.device, c_all_m_t.device, "|",
              s_img_t.device, s_split_t.device, s_all_m_t.device)
        print("[Info ] dtypes:",
              c_img_t.dtype, c_split_t.dtype, c_all_m_t.dtype, "|",
              s_img_t.dtype, s_split_t.dtype, s_all_m_t.dtype)

        # 与模型 dtype 对齐（通常 float32）
        model_dtype = next(self.makeup_model.parameters()).dtype

        def _to_model_dtype(t):
            return t.to(dtype=model_dtype, non_blocking=True) if t.dtype != model_dtype else t

        c_img_t   = _to_model_dtype(c_img_t)
        s_img_t   = _to_model_dtype(s_img_t)
        c_split_t = _to_model_dtype(c_split_t)
        s_split_t = _to_model_dtype(s_split_t)
        c_all_m_t = _to_model_dtype(c_all_m_t)
        s_all_m_t = _to_model_dtype(s_all_m_t)

        print("[Info ] final dtypes:",
              c_img_t.dtype, c_split_t.dtype, c_all_m_t.dtype, "|",
              s_img_t.dtype, s_split_t.dtype, s_all_m_t.dtype)

        data = {
            "non_makeup_color_img":   c_img_t,
            "non_makeup_split_parse": c_split_t,
            "non_makeup_all_mask":    c_all_m_t,
            "makeup_color_img":       s_img_t,
            "makeup_split_parse":     s_split_t,
            "makeup_all_mask":        s_all_m_t,
        }

        print("[Stage] model inference ...")
        out = self.makeup_model.test_pair(data)   # (B,C,H,W) in [-1,1]
        out = out[0].detach().float().cpu().numpy()
        out = np.transpose((out / 2 + 0.5) * 255.0, (1, 2, 0))
        out = np.clip(out + 0.5, 0, 255).astype(np.uint8)
        print("[Done ] inference, output:", out.shape, out.dtype)
        print("========== [CSDMTService.transfer DONE] ==========\n")
        return out
