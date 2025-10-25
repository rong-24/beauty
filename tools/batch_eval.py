#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
批量评测脚本
- 遍历 content_dir × style_dir
- RetouchFormer → CSD-MT → ACS自动调优
- 记录：耗时、α最优参数、指标（IDSim/StyleSim/SSIM/LPIPS）、ACS
- 存图 + CSV
"""

import os, sys, csv, time, traceback, warnings
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

# ===== 路径：把两个项目加入 Python 搜索路径 =====
sys.path.extend([
    "/root/autodl-tmp/beauty/retouchformer",
    "/root/autodl-tmp/beauty/CSD_MT"
])

# ===== 你的 in-process 类 =====
from retouch_infer.runner import RetouchFormerRunner
from csdmt_api import CSDMTService

# ===== 指标与融合依赖 =====
import cv2
from skimage.metrics import structural_similarity as ssim
try:
    import lpips
    _LPIPS_OK = True
except Exception as e:
    print("[Warn] lpips import failed -> fallback:", e)
    _LPIPS_OK = False

try:
    import clip
    _CLIP_OK = True
except Exception:
    _CLIP_OK = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True

# ---------------- I/O 辅助 ----------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def list_images(folder: Path) -> List[Path]:
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS])

def to_pil(arr_or_pil) -> Image.Image:
    if isinstance(arr_or_pil, Image.Image):
        return arr_or_pil.convert("RGB")
    arr = arr_or_pil
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L").convert("RGB")
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    return Image.fromarray(arr).convert("RGB")

# ---------------- Retouch / CSD-MT ----------------
def build_models(size: int = 512):
    print(f"[Init] Load models on {DEVICE} ...")
    rf = RetouchFormerRunner(
        model_name="RetouchFormer",
        ckpt_dir="/root/autodl-tmp/beauty/retouchformer/release_model",
        epoch="best",
        size=size,
        debug=False
    )
    cs = CSDMTService(
        csdmt_weights="/root/autodl-tmp/beauty/CSD_MT/CSD_MT/weights/CSD_MT.pth",
        faceparsing_weights="/root/autodl-tmp/beauty/CSD_MT/faceutils/face_parsing/res/cp/79999_iter.pth",
        device=DEVICE,
        align_mode="skip",
        resize=size
    )
    return rf, cs

def retouch_no_amp(rf: RetouchFormerRunner, pil_img: Image.Image) -> Image.Image:
    with torch.inference_mode():
        if hasattr(rf, "infer_image"):
            return rf.infer_image(pil_img)
        # 兜底：文件方式
        tmp_in = Path("/dev/shm/rf_in.jpg")
        tmp_out = Path("/dev/shm/rf_out.jpg")
        pil_img.save(tmp_in, quality=95)
        rf.infer_path(str(tmp_in), str(tmp_out))
        return Image.open(tmp_out).convert("RGB")

# ---------------- 融合与指标 ----------------
def rgb_to_lab(img_rgb_u8: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_rgb_u8, cv2.COLOR_RGB2LAB)

def lab_to_rgb(img_lab_u8: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(img_lab_u8, cv2.COLOR_LAB2RGB)
    return np.clip(rgb, 0, 255).astype(np.uint8)

def lowpass_ab(ab: np.ndarray, ksize=15, sigma=5):
    a = cv2.GaussianBlur(ab[..., 0], (ksize, ksize), sigma)
    b = cv2.GaussianBlur(ab[..., 1], (ksize, ksize), sigma)
    return np.stack([a, b], axis=-1)

def make_imperfection_map(I0_rgb: np.ndarray, Ir_rgb: np.ndarray, sigma=3) -> np.ndarray:
    # 保险：尺寸不同则把 Ir 对齐到 I0
    H, W = I0_rgb.shape[:2]
    if Ir_rgb.shape[:2] != (H, W):
        Ir_rgb = cv2.resize(Ir_rgb, (W, H), interpolation=cv2.INTER_AREA)

    I0 = (I0_rgb.astype(np.float32) / 255.0)
    Ir = (Ir_rgb.astype(np.float32) / 255.0)
    d = np.abs(Ir - I0).mean(axis=2)
    d = cv2.GaussianBlur(d, (0, 0), sigmaX=sigma)
    d = (d - d.min()) / (d.max() - d.min() + 1e-8)
    return d.astype(np.float32)


def cooperative_fusion(I0_rgb, Ir_rgb, Im_rgb, M_imp,
                       alpha_skin_base=0.7, alpha_lip=0.9, alpha_eye=0.8,
                       use_lowpass=True):
    Ir_lab = rgb_to_lab(Ir_rgb)
    Im_lab = rgb_to_lab(Im_rgb)
    Lr  = Ir_lab[..., 0:1]
    Abr = Ir_lab[..., 1:3].astype(np.float32)
    Abm = Im_lab[..., 1:3].astype(np.float32)
    Abm_use = lowpass_ab(Abm) if use_lowpass else Abm

    alpha_skin = np.clip(alpha_skin_base * (1.0 - M_imp), 0.0, 1.0)[..., None]
    Ab_skin = (1 - alpha_skin) * Abr + alpha_skin * Abm_use
    # 嘴唇/眼部区域掩码暂未接入 → 统一走皮肤融合即可（简化）
    Ab_final = Ab_skin

    out_lab  = np.concatenate([Lr, Ab_final], axis=-1)
    out_lab_u8 = np.clip(out_lab, 0, 255).astype(np.uint8)
    return lab_to_rgb(out_lab_u8)

# ---- Metrics ----
_LPIPS = None
def _get_lpips():
    if not _LPIPS_OK:
        raise RuntimeError("lpips not installed. pip install lpips")
    global _LPIPS
    if _LPIPS is None:
        _LPIPS = lpips.LPIPS(net="vgg").to(DEVICE).eval()
    return _LPIPS

def lpips_dist(im1_u8: np.ndarray, im2_u8: np.ndarray) -> float:
    if not _LPIPS_OK:
        return 0.5
    t1 = torch.from_numpy(im1_u8).permute(2,0,1).float().unsqueeze(0) / 255.0
    t2 = torch.from_numpy(im2_u8).permute(2,0,1).float().unsqueeze(0) / 255.0
    t1 = F.interpolate(t1, size=(256,256), mode="bilinear", align_corners=False)
    t2 = F.interpolate(t2, size=(256,256), mode="bilinear", align_corners=False)
    t1 = t1.to(DEVICE); t2 = t2.to(DEVICE)
    with torch.no_grad():
        d = _get_lpips()(t1*2-1, t2*2-1).item()
    return float(d)

def ssim_sim(im1_u8: np.ndarray, im2_u8: np.ndarray) -> float:
    if im1_u8.shape != im2_u8.shape:
        H, W = im1_u8.shape[:2]
        im2_u8 = cv2.resize(im2_u8, (W, H), interpolation=cv2.INTER_AREA)
    return float(ssim(im1_u8, im2_u8, channel_axis=2, data_range=255))


def clip_cosine(im1_u8: np.ndarray, im2_u8: np.ndarray):
    if not _CLIP_OK:
        return None
    try:
        model, preprocess = clip.load("ViT-B/32", device=DEVICE)  # 不传 download=
        with torch.no_grad():
            t1 = preprocess(Image.fromarray(im1_u8)).unsqueeze(0).to(DEVICE)
            t2 = preprocess(Image.fromarray(im2_u8)).unsqueeze(0).to(DEVICE)
            f1 = model.encode_image(t1); f2 = model.encode_image(t2)
            f1 = f1 / f1.norm(dim=-1, keepdim=True)
            f2 = f2 / f2.norm(dim=-1, keepdim=True)
            sim = (f1 @ f2.T).item()
        return float(sim)
    except Exception as e:
        print("[Warn] CLIP failed:", e)
        return None

def ab_style_similarity(Is_rgb: np.ndarray, If_rgb: np.ndarray) -> float:
    H, W = If_rgb.shape[:2]
    if Is_rgb.shape[:2] != (H, W):
        Is_rgb = cv2.resize(Is_rgb, (W, H), interpolation=cv2.INTER_AREA)
    Is_lab = rgb_to_lab(Is_rgb)
    If_lab = rgb_to_lab(If_rgb)
    a1 = Is_lab[...,1].astype(np.float32).reshape(-1)
    b1 = Is_lab[...,2].astype(np.float32).reshape(-1)
    a2 = If_lab[...,1].astype(np.float32).reshape(-1)
    b2 = If_lab[...,2].astype(np.float32).reshape(-1)
    v1 = np.stack([a1, b1], axis=0)
    v2 = np.stack([a2, b2], axis=0)
    num = (v1 * v2).sum()
    den = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8
    return float(num / den)

def compute_ACS(I0_rgb, Is_rgb, If_rgb, w_id=0.4, w_style=0.4, w_struct=0.2):
    lp = lpips_dist(I0_rgb, If_rgb)            # 小为好
    id_sim_lpips = 1.0 - max(0.0, min(1.0, lp))
    id_sim = id_sim_lpips
    cs = clip_cosine(I0_rgb, If_rgb)
    if cs is not None:
        id_sim = 0.5*id_sim_lpips + 0.5*cs
    style_sim  = ab_style_similarity(Is_rgb, If_rgb)
    struct_sim = ssim_sim(I0_rgb, If_rgb)
    acs = w_id * id_sim + w_style * style_sim + w_struct * struct_sim
    return float(acs), {"id_sim": id_sim, "style_sim": style_sim, "struct": struct_sim, "lpips": lp}

def auto_optimize_params(I0_rgb, Ir_rgb, Im_rgb, Is_rgb, M_imp,
                         grid_skin=(0.5, 0.7, 0.9),
                         grid_lip=(0.9,), grid_eye=(0.8,),
                         use_lowpass=True,
                         w_id=0.4, w_style=0.4, w_struct=0.2):
    best = {"score": -1e9, "params": None, "image": None, "metrics": None}
    for a_skin in grid_skin:
        for a_lip in grid_lip:
            for a_eye in grid_eye:
                If_fuse = cooperative_fusion(
                    I0_rgb, Ir_rgb, Im_rgb, M_imp,
                    alpha_skin_base=a_skin, alpha_lip=a_lip, alpha_eye=a_eye,
                    use_lowpass=use_lowpass
                )
                score, metrics = compute_ACS(I0_rgb, Is_rgb, If_fuse, w_id, w_style, w_struct)
                if score > best["score"]:
                    best.update({
                        "score": score,
                        "params": (a_skin, a_lip, a_eye, use_lowpass, w_id, w_style, w_struct),
                        "image": If_fuse,
                        "metrics": metrics
                    })
    return best

# ---------------- 主流程 ----------------
def run_batch(content_dir: Path, style_dir: Path, out_dir: Path,
              size=512, cs_amp=True,
              w_id=0.4, w_style=0.4, w_struct=0.2,
              grid_skin=(0.5,0.7,0.9), grid_lip=(0.9,), grid_eye=(0.8,),
              use_lowpass=True):

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "summary.csv"
    err_log  = open(out_dir / "errors.log", "a")

    # 写表头
    header = ["index","content","style",
              "t_retouch","t_csdmt",
              "alpha_skin_base","alpha_lip","alpha_eye","lowpass",
              "lambda_id","lambda_style","lambda_struct",
              "IDSim","StyleSim","SSIM","LPIPS","ACS","save_dir"]
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(header)

    rf, cs = build_models(size=size)

    contents = list_images(content_dir)
    styles   = list_images(style_dir)

    idx = 0
    for cpath in contents:
        for spath in styles:
            idx += 1
            pair_dir = out_dir / f"{cpath.stem}__{spath.stem}"
            pair_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n[{idx}] {cpath.name} × {spath.name}")

            try:
                I0 = to_pil(Image.open(cpath).convert("RGB"))
                Is = to_pil(Image.open(spath).convert("RGB"))

                # Retouch
# ================= Retouch（方法一 + 方法二 + 自检）=================
                t0 = time.time()

                # 方法一：放大后再修饰（让小斑点在网络里更“显眼”）
                RETOUCH_SCALE = 1.5  # 可调：1.5~2.0（GPU 允许的话 2.0 更猛）
                new_w = max(1, int(I0.width  * RETOUCH_SCALE))
                new_h = max(1, int(I0.height * RETOUCH_SCALE))
                I0_up = I0.resize((new_w, new_h), Image.BICUBIC)

                # 调用 RetouchFormer（与你的原逻辑一致）
                Ir_up = retouch_no_amp(rf, I0_up)  # 若你的函数签名是 (rf, img, pair_dir)，这里传 pair_dir

                # 缩回到原图尺寸
                Ir = Ir_up.resize(I0.size, Image.BICUBIC)

                # 方法二：对“修饰残差”做可控增益，增强可见度但不过度
                RETOUCH_GAIN = 1.25  # 可调：1.1~1.4；1.0 等于不增益
                if abs(RETOUCH_GAIN - 1.0) > 1e-6:
                    I0_np = np.array(I0, dtype=np.float32)
                    Ir_np = np.array(Ir, dtype=np.float32)
                    Ir_np = np.clip(I0_np + RETOUCH_GAIN * (Ir_np - I0_np), 0, 255).astype(np.uint8)
                    Ir = Image.fromarray(Ir_np)

                t1 = time.time()

                # —— 自检：Retouch 是否“真的动了图”（便于你快速判断是否需要调大 scale/gain）
                I0_np_chk = np.array(I0, dtype=np.float32)
                Ir_np_chk = np.array(Ir, dtype=np.float32)
                mad = float(np.mean(np.abs(Ir_np_chk - I0_np_chk)))  # 平均绝对像素差（0~255）
                try:
                    lp_rf = lpips_dist(I0_np_chk.astype(np.uint8), Ir_np_chk.astype(np.uint8)) if _LPIPS_OK else -1.0
                except Exception as _e:
                    print("[Warn] LPIPS on Retouch failed:", _e)
                    lp_rf = -1.0

                if mad < 2.0 and (lp_rf < 0.02 or lp_rf < 0):
                    # 变化极小，提醒你可以把 RETOUCH_SCALE/GAIN 再调大一点
                    print(f"    [Warn] Retouch 变化较小 (MAD={mad:.2f}, LPIPS={lp_rf:.4f}) -> 可尝试提高 RETOUCH_SCALE/GAIN")
                else:
                    print(f"    [RF] scale={RETOUCH_SCALE}, gain={RETOUCH_GAIN} -> MAD={mad:.2f}, LPIPS={lp_rf:.4f}")

                # （保持你后面的尺寸对齐检查也没问题；上面已经把 Ir 缩回 I0.size 了）
                # if Ir.size != I0.size:
                #     Ir = Ir.resize(I0.size, Image.BICUBIC)



                    # CSD-MT
                    with torch.inference_mode():
                        if cs_amp:
                            with torch.cuda.amp.autocast(dtype=torch.float16):
                                Im_np = cs.transfer(Ir, Is)
                        else:
                            Im_np = cs.transfer(Ir, Is)
                    t2 = time.time()

                    I0_np = np.array(I0); Is_np = np.array(Is)
                    Ir_np = np.array(Ir); Im_np = np.array(Im_np)
                    M_imp = make_imperfection_map(I0_np, Ir_np, sigma=3)

                    H, W = I0_np.shape[:2]
                    if Im_np.shape[:2] != (H, W):
                        Im_np = cv2.resize(Im_np, (W, H), interpolation=cv2.INTER_AREA)

                # 自动调优
                best = auto_optimize_params(
                    I0_np, Ir_np, Im_np, Is_np, M_imp,
                    grid_skin=grid_skin, grid_lip=grid_lip, grid_eye=grid_eye,
                    use_lowpass=use_lowpass,
                    w_id=w_id, w_style=w_style, w_struct=w_struct
                )
                a_skin, a_lip, a_eye, lpflag, w_id_, w_style_, w_struct_ = best["params"]
                score = best["score"]; metrics = best["metrics"]
                If_opt = best["image"]

                # 存图
                I0.save(pair_dir / "content.jpg", quality=95)
                Is.save(pair_dir / "style.jpg",   quality=95)
                Ir.save(pair_dir / "retouch.jpg", quality=95)
                Image.fromarray(Im_np).save(pair_dir / "csdmt.jpg", quality=95)
                Image.fromarray(If_opt).save(pair_dir / "optimized.jpg", quality=95)

                # 写CSV
                row = [idx, cpath.name, spath.name,
                       round(t1-t0,3), round(t2-t1,3),
                       a_skin, a_lip, a_eye, int(lpflag),
                       w_id_, w_style_, w_struct_,
                       round(metrics["id_sim"],4), round(metrics["style_sim"],4),
                       round(metrics["struct"],4), round(metrics["lpips"],4),
                       round(score,4), str(pair_dir)]
                with open(csv_path, "a", newline="") as f:
                    csv.writer(f).writerow(row)

                print(f"  -> Done. ACS={score:.4f}, α=({a_skin},{a_lip},{a_eye}), t={t2-t0:.2f}s")
            except Exception as e:
                msg = f"[ERROR] {cpath.name} × {spath.name} :: {e}\n{traceback.format_exc()}\n"
                print(msg)
                err_log.write(msg); err_log.flush()
                continue

    err_log.close()
    print(f"\n[Finish] CSV: {csv_path}")
    print(f"[Finish] Images saved under: {out_dir}")

# ---------------- CLI ----------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--content_dir", default="/root/autodl-tmp/beauty/data/content")
    ap.add_argument("--style_dir",   default="/root/autodl-tmp/beauty/data/style")
    ap.add_argument("--out_dir",     default="/root/autodl-tmp/beauty/outputs_batch")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--no_amp", action="store_true", help="关闭 CSD-MT AMP 半精度")
    ap.add_argument("--w_id",     type=float, default=0.4)
    ap.add_argument("--w_style",  type=float, default=0.4)
    ap.add_argument("--w_struct", type=float, default=0.2)
    ap.add_argument("--grid_skin", default="0.5,0.7,0.9", help="逗号分隔")
    ap.add_argument("--grid_lip",  default="0.9",         help="逗号分隔（简化，先固定）")
    ap.add_argument("--grid_eye",  default="0.8",         help="逗号分隔（简化，先固定）")
    ap.add_argument("--lowpass",   action="store_true", help="启用低通色彩融合")
    args = ap.parse_args()

    gs = tuple(float(x) for x in args.grid_skin.split(",") if x.strip())
    gl = tuple(float(x) for x in args.grid_lip.split(",")  if x.strip())
    ge = tuple(float(x) for x in args.grid_eye.split(",")  if x.strip())

    run_batch(
        content_dir=Path(args.content_dir),
        style_dir=Path(args.style_dir),
        out_dir=Path(args.out_dir),
        size=args.size,
        cs_amp=(not args.no_amp),
        w_id=args.w_id, w_style=args.w_style, w_struct=args.w_struct,
        grid_skin=gs, grid_lip=gl, grid_eye=ge,
        use_lowpass=args.lowpass
    )
