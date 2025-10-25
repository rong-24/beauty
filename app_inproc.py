#!/usr/bin/env python3
import os, sys, time, traceback
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import gradio as gr

# ===== 追加依赖（ACS / 融合所需） =====
import cv2
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
try:
    import lpips
    _LPIPS_OK = True
except Exception as e:
    print("[Warn] lpips import failed:", e)
    _LPIPS_OK = False

# 可选：CLIP，没装或没网也能跑
try:
    import clip
    _CLIP_OK = True
except Exception:
    _CLIP_OK = False

# ===== 在 Retouchformer 环境里跑 =====
sys.path.extend([
    "/root/autodl-tmp/beauty/retouchformer",
    "/root/autodl-tmp/beauty/CSD_MT"
])

assert torch.cuda.is_available(), "未检测到 CUDA，请在 GPU 环境运行。"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
torch.backends.cudnn.benchmark = True
try:
    torch.set_default_device("cuda")  # 对 factory 有效
except Exception:
    pass

DEVICE = "cuda"
DEFAULT_SIZE = 512

from retouch_infer.runner import RetouchFormerRunner
from csdmt_api import CSDMTService

print(f"[Init] Loading RetouchFormer on {DEVICE} ...")
rf = RetouchFormerRunner(
    model_name="RetouchFormer",
    ckpt_dir="/root/autodl-tmp/beauty/retouchformer/release_model",
    epoch="best",
    size=DEFAULT_SIZE,
    debug=True   # 开打印，便于定位
)

print(f"[Init] Loading CSD-MT on {DEVICE} ...")
cs = CSDMTService(
    csdmt_weights="/root/autodl-tmp/beauty/CSD_MT/CSD_MT/weights/CSD_MT.pth",
    faceparsing_weights="/root/autodl-tmp/beauty/CSD_MT/faceutils/face_parsing/res/cp/79999_iter.pth",
    device=DEVICE,
    align_mode="skip",
    resize=DEFAULT_SIZE
)

# =================== 工具：安全包装（打印 traceback + 前端报错） ===================
def safe_call(fn):
    def _wrap(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except gr.Error:
            raise
        except Exception as e:
            print("\n[ERROR]", e)
            traceback.print_exc()
            raise gr.Error(f"后端异常：{e.__class__.__name__}: {e}")
    return _wrap

# =================== I/O 辅助 ===================
def _np_to_pil(arr: np.ndarray) -> Image.Image:
    if arr is None:
        raise gr.Error("未接收到图像。")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L").convert("RGB")
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    return Image.fromarray(arr).convert("RGB")

# =================== Retouch（保持你原逻辑） ===================
def retouch_no_amp(pil_img: Image.Image) -> Image.Image:
    # 统一关闭 AMP（部分激活函数对 AMP 不友好）
    if hasattr(rf, "infer_image"):
        with torch.inference_mode():
            return rf.infer_image(pil_img)
    # 回退：临时文件 + infer_path
    tmp_in  = Path("/dev/shm/input_rf.jpg")
    tmp_out = Path("/dev/shm/output_rf.jpg")
    pil_img.save(tmp_in, format="JPEG", quality=95)
    with torch.inference_mode():
        rf.infer_path(str(tmp_in), str(tmp_out))
    return Image.open(tmp_out).convert("RGB")

# =================== 路线B：ACS & 协同融合（无掩码版，可后续接入mask） ===================
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
    I0 = (I0_rgb.astype(np.float32) / 255.0)
    Ir = (Ir_rgb.astype(np.float32) / 255.0)
    d = np.abs(Ir - I0).mean(axis=2)
    d = cv2.GaussianBlur(d, (0, 0), sigmaX=sigma)
    d = (d - d.min()) / (d.max() - d.min() + 1e-8)
    return d.astype(np.float32)  # [H,W], 0~1

def cooperative_fusion(I0_rgb, Ir_rgb, Im_rgb, M_imp,
                       M_skin=None, M_lip=None, M_eye=None,
                       alpha_skin_base=0.7, alpha_lip=0.9, alpha_eye=0.8,
                       use_lowpass=True):
    # 不改亮度，保留 RF 细节；AB 做融合
    Ir_lab = rgb_to_lab(Ir_rgb)
    Im_lab = rgb_to_lab(Im_rgb)
    Lr  = Ir_lab[..., 0:1]
    Abr = Ir_lab[..., 1:3].astype(np.float32)
    Abm = Im_lab[..., 1:3].astype(np.float32)
    Abm_use = lowpass_ab(Abm) if use_lowpass else Abm

    H, W = Lr.shape[:2]
    one  = np.ones((H, W), np.float32)
    zero = np.zeros((H, W), np.float32)

    if M_skin is None: M_skin = one
    if M_lip  is None: M_lip  = zero
    if M_eye  is None: M_eye  = zero

    alpha_skin = np.clip(alpha_skin_base * (1.0 - M_imp), 0.0, 1.0)

    Ms = M_skin[..., None]
    Ml = M_lip[..., None]
    Me = M_eye[..., None]
    Mr = (1.0 - np.clip(M_skin + M_lip + M_eye, 0.0, 1.0))[..., None]

    Ab_skin = (1 - alpha_skin[..., None]) * Abr + alpha_skin[..., None] * Abm_use
    Ab_lip  = (1 - alpha_lip) * Abr + alpha_lip * Abm
    Ab_eye  = (1 - alpha_eye) * Abr + alpha_eye * Abm

    Ab_final = Ab_skin * Ms + Ab_lip * Ml + Ab_eye * Me + Abr * Mr
    out_lab  = np.concatenate([Lr, Ab_final], axis=-1)
    out_lab_u8 = np.clip(out_lab, 0, 255).astype(np.uint8)
    out_rgb  = lab_to_rgb(out_lab_u8)
    return out_rgb

# ----- 指标：LPIPS / SSIM / 可选CLIP -----
_LPIPS = None
def _get_lpips():
    if not _LPIPS_OK:
        raise RuntimeError("lpips 未安装：pip install lpips")
    global _LPIPS
    if _LPIPS is None:
        _LPIPS = lpips.LPIPS(net='vgg').to(DEVICE).eval()
    return _LPIPS

def lpips_dist(im1_u8: np.ndarray, im2_u8: np.ndarray) -> float:
    if not _LPIPS_OK:
        return 0.5  # 兜底：没有lpips时，给个中性值，避免崩溃
    t1 = torch.from_numpy(im1_u8).permute(2,0,1).float().unsqueeze(0) / 255.0
    t2 = torch.from_numpy(im2_u8).permute(2,0,1).float().unsqueeze(0) / 255.0
    t1 = F.interpolate(t1, size=(256,256), mode='bilinear', align_corners=False)
    t2 = F.interpolate(t2, size=(256,256), mode='bilinear', align_corners=False)
    t1 = t1.to(DEVICE); t2 = t2.to(DEVICE)
    with torch.no_grad():
        d = _get_lpips()(t1*2-1, t2*2-1).item()
    return float(d)

def ssim_sim(im1_u8: np.ndarray, im2_u8: np.ndarray) -> float:
    return float(ssim(im1_u8, im2_u8, channel_axis=2, data_range=255))

def clip_cosine(im1_u8: np.ndarray, im2_u8: np.ndarray):
    """
    可选的身份相似（CLIP 图像嵌入余弦）。失败时返回 None，不影响主流程。
    """
    if not _CLIP_OK:
        return None
    try:
        # 某些版本的 clip.load 不支持 download 参数
        model, preprocess = clip.load("ViT-B/32", device=DEVICE)  # 去掉 download=
        with torch.no_grad():
            t1 = preprocess(Image.fromarray(im1_u8)).unsqueeze(0).to(DEVICE)
            t2 = preprocess(Image.fromarray(im2_u8)).unsqueeze(0).to(DEVICE)
            f1 = model.encode_image(t1); f2 = model.encode_image(t2)
            f1 = f1 / f1.norm(dim=-1, keepdim=True)
            f2 = f2 / f2.norm(dim=-1, keepdim=True)
            sim = (f1 @ f2.T).item()
        return float(sim)
    except Exception as e:
        print("[Warn] CLIP similarity failed:", e)
        return None

def ab_style_similarity(Is_rgb: np.ndarray, If_rgb: np.ndarray) -> float:
    """
    计算风格色彩相似：把风格图 Is resize 到 If 的分辨率后，
    在 LAB 的 AB 通道做余弦相似度（越大越相似）。
    """
    H, W = If_rgb.shape[:2]
    if Is_rgb.shape[:2] != (H, W):
        Is_resized = cv2.resize(Is_rgb, (W, H), interpolation=cv2.INTER_AREA)
    else:
        Is_resized = Is_rgb

    Is_lab = cv2.cvtColor(Is_resized, cv2.COLOR_RGB2LAB)
    If_lab = cv2.cvtColor(If_rgb,     cv2.COLOR_RGB2LAB)

    a1 = Is_lab[..., 1].astype(np.float32).reshape(-1)
    b1 = Is_lab[..., 2].astype(np.float32).reshape(-1)
    a2 = If_lab[..., 1].astype(np.float32).reshape(-1)
    b2 = If_lab[..., 2].astype(np.float32).reshape(-1)

    v1 = np.stack([a1, b1], axis=0)
    v2 = np.stack([a2, b2], axis=0)

    num = (v1 * v2).sum()
    den = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8
    return float(num / den)

def compute_ACS(I0_rgb, Is_rgb, If_rgb, w_id=0.4, w_style=0.4, w_struct=0.2):
    # IDSim：用 1-LPIPS（可选与CLIP平均）
    lp = lpips_dist(I0_rgb, If_rgb)             # 距离：小好
    id_sim_lpips = 1.0 - max(0.0, min(1.0, lp))
    id_sim = id_sim_lpips
    cs = clip_cosine(I0_rgb, If_rgb)
    if cs is not None:
        id_sim = 0.5*id_sim_lpips + 0.5*cs
    style_sim  = ab_style_similarity(Is_rgb, If_rgb)  # 大好
    struct_sim = ssim_sim(I0_rgb, If_rgb)             # 大好
    acs = w_id * id_sim + w_style * style_sim + w_struct * struct_sim
    return float(acs), {"id_sim": id_sim, "style_sim": style_sim, "struct": struct_sim, "lpips": lp}

def auto_optimize_params(I0_rgb, Ir_rgb, Im_rgb, Is_rgb, M_imp,
                         grid_skin=(0.5, 0.7, 0.9),
                         grid_lip=(0.8, 0.9, 1.0),
                         grid_eye=(0.7, 0.8, 0.9),
                         use_lowpass=True,
                         w_id=0.4, w_style=0.4, w_struct=0.2):
    best = {"score": -1e9, "params": None, "image": None, "metrics": None}
    for a_skin in grid_skin:
        for a_lip in grid_lip:
            for a_eye in grid_eye:
                If_fuse = cooperative_fusion(
                    I0_rgb, Ir_rgb, Im_rgb, M_imp,
                    M_skin=None, M_lip=None, M_eye=None,  # 先不接mask，后续可替换
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

# =================== 原有主流程 ===================
@safe_call
def pipeline(content_np, style_np, size=DEFAULT_SIZE, cs_amp=True):
    if content_np is None or style_np is None:
        raise gr.Error("请同时提供【原图】与【参考妆容图】。")

    size = int(size)
    t0 = time.time()
    c_img = _np_to_pil(content_np)
    s_img = _np_to_pil(style_np)

    # Step1: Retouch（禁 AMP）
    r0 = time.time()
    retouched_img = retouch_no_amp(c_img)
    r1 = time.time()

    # Step2: CSD-MT（可选 AMP）
    m0 = time.time()
    if cs_amp:
        with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
            final_np = cs.transfer(retouched_img, s_img)
    else:
        with torch.inference_mode():
            final_np = cs.transfer(retouched_img, s_img)
    m1 = time.time()

    total = time.time() - t0
    stats = (
        f"总耗时: {total:.2f}s\n"
        f"  • RetouchFormer(AMP关): {r1 - r0:.2f}s\n"
        f"  • CSD-MT(AMP={'开' if cs_amp else '关'}): {m1 - m0:.2f}s\n"
        f"提示：若仍报 Float vs cuda，检查 csdmt_api.py 的 'pack tensors' 是否 .to(self.device)。"
    )
    return np.array(c_img), np.array(s_img), np.array(retouched_img), final_np, stats

# =================== 自动调优入口（用现成四图做后处理+打分） ===================
@safe_call
def auto_optimize_entry(I0_np, Is_np, Ir_np, If_np,
                        a_skin_list, a_lip_list, a_eye_list, use_lowpass,
                        w_id, w_style, w_struct):
    if any(x is None for x in [I0_np, Is_np, Ir_np, If_np]):
        raise gr.Error("请先点击“开始处理”得到四张图，再进行自动优化。")

    try:
        a_skin = [float(x) for x in (a_skin_list if a_skin_list else ["0.7","0.9"])]
        a_lip  = [float(x) for x in (a_lip_list  if a_lip_list  else ["0.9","1.0"])]
        a_eye  = [float(x) for x in (a_eye_list  if a_eye_list  else ["0.8"])]
        w_id   = float(w_id); w_style=float(w_style); w_struct=float(w_struct)
    except Exception as e:
        print("[Parse] 参数解析失败：", e)
        raise gr.Error("参数格式不正确，请检查。")

    # 改动强度图（基于 I0 vs Ir）
    M_imp = make_imperfection_map(I0_np, Ir_np, sigma=3)

    best = auto_optimize_params(
        I0_rgb=I0_np, Ir_rgb=Ir_np, Im_rgb=If_np, Is_rgb=Is_np, M_imp=M_imp,
        grid_skin=tuple(a_skin), grid_lip=tuple(a_lip), grid_eye=tuple(a_eye),
        use_lowpass=bool(use_lowpass),
        w_id=w_id, w_style=w_style, w_struct=w_struct
    )
    a_skin, a_lip, a_eye, lpflag, w_id, w_style, w_struct = best["params"]
    msg = (
        f"最优ACS: {best['score']:.4f}\n"
        f"参数: alpha_skin_base={a_skin}, alpha_lip={a_lip}, alpha_eye={a_eye}, lowpass={lpflag}\n"
        f"权重: λ1(w_id)={w_id}, λ2(w_style)={w_style}, λ3(w_struct)={w_struct}\n"
        f"子项: IDSim={best['metrics']['id_sim']:.4f}, StyleSim={best['metrics']['style_sim']:.4f}, "
        f"SSIM={best['metrics']['struct']:.4f}, LPIPS={best['metrics']['lpips']:.4f}"
    )
    return best["image"], msg

# =================== UI ===================
with gr.Blocks(title="同进程：RetouchFormer ➜ CSD-MT（Retouchformer 环境）") as demo:
    gr.Markdown("### 同进程常驻 + 统一 CUDA（Retouch AMP 关闭，CSD-MT 可选 AMP）")
    with gr.Row():
        with gr.Column():
            content = gr.Image(type="numpy", label="原图（Non-makeup）", height=320)
            style   = gr.Image(type="numpy", label="参考妆容（Makeup）",   height=320)
            size    = gr.Slider(256, 768, value=DEFAULT_SIZE, step=32, label="分辨率（如需动态改，后续我给你补 setter）")
            cs_amp  = gr.Checkbox(value=True, label="CSD-MT 开启半精度 AMP")
            run_btn = gr.Button("✨ 开始处理", variant="primary")
        with gr.Column():
            out_c   = gr.Image(label="原图", type="numpy")
            out_s   = gr.Image(label="参考妆容", type="numpy")
            out_r   = gr.Image(label="修饰图（RetouchFormer）", type="numpy")
            out_f   = gr.Image(label="成品图（CSD-MT）", type="numpy")
            stats   = gr.Textbox(label="性能统计", lines=7)

    run_btn.click(fn=pipeline, inputs=[content, style, size, cs_amp], outputs=[out_c, out_s, out_r, out_f, stats])

    gr.Markdown("---")
    gr.Markdown("### 🔧 自动调优（ACS 驱动）")
    with gr.Row():
        a_skin_list = gr.CheckboxGroup(choices=["0.5","0.6","0.7","0.8","0.9"], value=["0.7","0.9"], label="alpha_skin_base 候选")
        a_lip_list  = gr.CheckboxGroup(choices=["0.8","0.9","1.0"], value=["0.9","1.0"], label="alpha_lip 候选")
        a_eye_list  = gr.CheckboxGroup(choices=["0.7","0.8","0.9"], value=["0.8"], label="alpha_eye 候选")
        use_lowpass = gr.Checkbox(value=True, label="皮肤色彩低通（更自然）")
    with gr.Row():
        w_id     = gr.Number(value=0.4, label="λ1 = w_id（身份保持）")
        w_style  = gr.Number(value=0.4, label="λ2 = w_style（妆容匹配）")
        w_struct = gr.Number(value=0.2, label="λ3 = w_struct（结构相似）")
    auto_btn = gr.Button("🚀 一键自动优化（ACS）", variant="secondary")
    opt_img = gr.Image(label="自动优化结果", type="numpy")
    opt_msg = gr.Textbox(label="优化结果与参数", lines=6)

    auto_btn.click(
        fn=auto_optimize_entry,
        inputs=[out_c, out_s, out_r, out_f, a_skin_list, a_lip_list, a_eye_list, use_lowpass, w_id, w_style, w_struct],
        outputs=[opt_img, opt_msg]
    )

demo.launch(server_name="0.0.0.0", server_port=7860)
