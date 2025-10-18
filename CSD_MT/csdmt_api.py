# csdmt_api.py
# -*- coding: utf-8 -*-

# ========= 基础路径与导入修正 =========
import os, sys
HERE = os.path.abspath(os.path.dirname(__file__))  # 当前脚本所在目录
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import os.path as osp
from typing import Optional, Union
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

# ========= 依赖 CSD-MT 主模型与配置 =========
from CSD_MT.options import Options
from CSD_MT.model import CSD_MT

# ========= 人脸解析（BiSeNet，不依赖 dlib）=========
from faceutils.face_parsing.model import BiSeNet

ArrayLike = Union[np.ndarray, Image.Image]


class CSDMTService:
    """
    轻量可嵌入的 CSD-MT 推理封装。
    - 默认 align_mode='skip'：不做 dlib 对齐（需保证人脸大致居中/无遮挡）。
    - 如需 dlib 对齐：align_mode='dlib'（需要你安装 dlib，并保留 faceutils/dlibutils）。
    """

    def __init__(
        self,
        csdmt_weights: str = osp.join(HERE, "CSD_MT/weights/CSD_MT.pth"),
        faceparsing_weights: str = osp.join(HERE, "faceutils/face_parsing/res/cp/79999_iter.pth"),
        device: Optional[str] = None,
        align_mode: str = "skip",   # 'skip' | 'dlib'
        resize: int = 256,
    ):
        print("========== [CSDMTService.__init__] ==========")
        # 设备
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.resize = resize
        self.align_mode = align_mode
        print(f"[Init] device        : {self.device}")
        print(f"[Init] resize        : {self.resize}")
        print(f"[Init] align_mode    : {self.align_mode}")
        print(f"[Init] CSD-MT weights: {csdmt_weights}")
        print(f"[Init] FP  weights   : {faceparsing_weights}")

        # 1) 加载 BiSeNet(19 类)
        print("[Init] Loading BiSeNet (face parsing) ...")
        self.n_classes = 19
        self.face_parsing = BiSeNet(n_classes=self.n_classes)
        try:
            # 如果权重是纯 state_dict，这里 OK；若是完整 ckpt，同样可加载
            state = torch.load(faceparsing_weights, map_location="cpu")
            self.face_parsing.load_state_dict(state)
        except Exception as e:
            print("[FATAL] Failed to load face parsing weights!")
            print("        Path:", faceparsing_weights)
            raise
        self.face_parsing.to(self.device).eval()
        print("[Init] BiSeNet loaded and set to eval().")

        # 2) 加载 CSD-MT 主模型
        print("[Init] Parsing CSD-MT options (detached from CLI) ...")
        parser = Options()
        # 关键：避免和外部 argparse 冲突（比如 --out 被误认为 --output_dim）
        _argv_backup = sys.argv[:]  # 复制一份
        try:
            sys.argv = [sys.argv[0]]  # 只保留程序名，相当于“无参数”
            self.opts = parser.parse()  # 你这个实现不接受 args，因此直接调用
        finally:
            sys.argv = _argv_backup     # 恢复 sys.argv

        # 强制与外部 resize 对齐
        self.opts.resize_size = resize
        print(f"[Init] opts.resize_size => {self.opts.resize_size}")

        print("[Init] Building CSD-MT model ...")
        self.makeup_model = CSD_MT(self.opts)
        try:
            _ = self.makeup_model.resume(csdmt_weights)  # 恢复权重
        except Exception as e:
            print("[FATAL] Failed to load CSD-MT weights!")
            print("        Path:", csdmt_weights)
            raise
        self.makeup_model.to(self.device).eval()
        print("[Init] CSD-MT loaded and set to eval().")

        # 3) 预处理/后处理
        self._to_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        print("========== [CSDMTService.__init__ DONE] ==========\n")

    # --------------- 可选对齐（默认跳过 dlib） ---------------
    def _maybe_align(self, image: ArrayLike) -> np.ndarray:
        """
        align_mode='skip'：不做对齐，原样（转为 RGB np.uint8）。
        align_mode='dlib'：使用 faceutils.dlibutils 做检测/裁剪（需要 dlib）。
        """
        if isinstance(image, Image.Image):
            img = image.convert("RGB")
        else:
            img = Image.fromarray(image).convert("RGB")

        if self.align_mode == "skip":
            return np.array(img)

        if self.align_mode == "dlib":
            print("[Align] Using dlib alignment ...")
            # 惰性导入，避免顶层 import dlib
            try:
                from faceutils import get_dlib
                futils = get_dlib()
            except Exception as e:
                print("[FATAL] dlib alignment requested, but faceutils/dlibutils unavailable.")
                raise
            up_ratio   = 0.2 / 0.85
            down_ratio = 0.15 / 0.85
            width_ratio= 0.2 / 0.85
            faces = futils.detect(img)
            if not faces:
                raise ValueError("No face detected for dlib alignment.")
            image_aligned, _, _ = futils.crop(img, faces[0], up_ratio, down_ratio, width_ratio)
            return np.array(image_aligned)

        raise ValueError(f"Unknown align_mode: {self.align_mode}")

    # --------------- 人脸解析 → 19类mask ---------------
    @torch.no_grad()
    def _face_parsing_19(self, x: np.ndarray) -> np.ndarray:
        img = Image.fromarray(x).resize((512, 512), Image.BILINEAR)
        ten = self._to_tensor(img).unsqueeze(0).to(self.device)
        out = self.face_parsing(ten)[0]              # (B=1, C=19, H=512, W=512)
        parsing = out.squeeze(0).argmax(0).detach().to("cpu").numpy().astype(np.int32)  # (H,W)
        return parsing

    # --------------- 19类 → CSD-MT 语义分路（与原逻辑一致） ---------------
    def _split_parse(self, parse: np.ndarray) -> np.ndarray:
        h, w = parse.shape
        c = self.opts.semantic_dim
        result = np.zeros([h, w, c], dtype=np.float32)

        # 映射关系（与 quick_start/CSD_MT_eval.py 一致）
        # 背景/耳朵/颈/衣领/鼻：0,16,17,18,9
        result[:, :, 0][np.isin(parse, [0, 16, 17, 18, 9])] = 1
        # 眉：1,6
        result[:, :, 1][np.isin(parse, [1, 6])] = 1
        # 眼：2,3
        result[:, :, 2][np.isin(parse, [2, 3])] = 1
        # 眼影：4,5
        result[:, :, 3][np.isin(parse, [4, 5])] = 1
        # 鼻梁/下巴：7,8
        result[:, :, 4][np.isin(parse, [7, 8])] = 1
        # 上唇：10
        result[:, :, 5][parse == 10] = 1
        # 下唇：11
        result[:, :, 6][parse == 11] = 1
        # 牙齿：12
        result[:, :, 7][parse == 12] = 1
        # 皮肤：13
        result[:, :, 8][parse == 13] = 1
        # 头发/帽子：14,15
        result[:, :, 9][np.isin(parse, [14, 15])] = 1
        return result

    # --------------- 生成全局/局部 mask（与原逻辑一致） ---------------
    def _local_masks(self, split_parse: np.ndarray) -> np.ndarray:
        h, w, _ = split_parse.shape
        all_mask = np.zeros([h, w], dtype=np.float32)
        all_mask[split_parse[:, :, 0] == 0] = 1
        all_mask[split_parse[:, :, 3] == 1] = 0
        all_mask[split_parse[:, :, 6] == 1] = 0
        all_mask = np.repeat(all_mask[:, :, None], 3, axis=2)
        return all_mask

    # --------------- 主入口：做一次妆容迁移 ---------------
    @torch.no_grad()
    def transfer(self, content: ArrayLike, style: ArrayLike) -> np.ndarray:
        print("========== [CSDMTService.transfer] ==========")
        # 1) 对齐（或跳过）
        print("[Stage] maybe_align(content) ...")
        c_img = self._maybe_align(content)
        print("[Info ] content after align:", c_img.shape, c_img.dtype)

        print("[Stage] maybe_align(style) ...")
        s_img = self._maybe_align(style)
        print("[Info ] style   after align:", s_img.shape, s_img.dtype)

        # 2) resize
        print(f"[Stage] resize to {self.resize}x{self.resize} ...")
        c_img = np.array(Image.fromarray(c_img).resize((self.resize, self.resize), Image.BILINEAR))
        s_img = np.array(Image.fromarray(s_img).resize((self.resize, self.resize), Image.BILINEAR))
        print("[Info ] content resized:", c_img.shape, c_img.dtype)
        print("[Info ] style   resized:", s_img.shape, s_img.dtype)

        # 3) 解析 → 语义分路/全局 mask
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

        # 4) 归一化到[-1,1] + CHW
        print("[Stage] pack tensors ...")
        def prep(img, split, all_m):
            img = (img.astype(np.float32) / 127.5) - 1.0
            img = np.transpose(img, (2, 0, 1))          # HWC -> CHW
            split = np.transpose(split, (2, 0, 1))
            all_m = np.transpose(all_m, (2, 0, 1))
            return (
                torch.from_numpy(img)[None].to(self.device),
                torch.from_numpy(split)[None].to(self.device),
                torch.from_numpy(all_m)[None].to(self.device),
            )

        c_img_t, c_split_t, c_all_m_t = prep(c_img, c_split, c_all_m)
        s_img_t, s_split_t, s_all_m_t = prep(s_img, s_split, s_all_m)
        print("[Info ] Tensors shapes:",
              "c_img", tuple(c_img_t.shape),
              "c_split", tuple(c_split_t.shape),
              "c_all_m", tuple(c_all_m_t.shape),
              "| s_img", tuple(s_img_t.shape),
              "s_split", tuple(s_split_t.shape),
              "s_all_m", tuple(s_all_m_t.shape))

        data = {
            "non_makeup_color_img": c_img_t,
            "non_makeup_split_parse": c_split_t,
            "non_makeup_all_mask": c_all_m_t,
            "makeup_color_img": s_img_t,
            "makeup_split_parse": s_split_t,
            "makeup_all_mask": s_all_m_t,
        }

        # 5) 推理
        print("[Stage] model inference ...")
        out = self.makeup_model.test_pair(data)  # (B,C,H,W), [-1,1]
        out = out[0].detach().float().cpu().numpy()
        out = np.transpose((out / 2 + 0.5) * 255.0, (1, 2, 0))
        out = np.clip(out + 0.5, 0, 255).astype(np.uint8)
        print("[Done ] inference, output:", out.shape, out.dtype)
        print("========== [CSDMTService.transfer DONE] ==========\n")
        return out
