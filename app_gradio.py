#!/usr/bin/env python3
import os, io, shutil, tempfile, subprocess, time
from pathlib import Path

import numpy as np
from PIL import Image
import gradio as gr

# ========= 配置（按需修改） =========
CSDMT_ENV = "CSDMT"                    # 你的 CSDMT conda 环境名
RETOUCH_ENV = "Retouchformer"          # 你的 Retouchformer conda 环境名

CSDMT_DIR = Path("/root/autodl-tmp/beauty/CSD_MT")
RETOUCH_DIR = Path("/root/autodl-tmp/beauty/retouchformer")

CKPT_DIR = RETOUCH_DIR / "release_model"  # RetouchFormer 模型目录
EPOCH = "best"

# 示例目录（可改为你自己的）
EX_NON = CSDMT_DIR / "examples/non_makeup"
EX_MK  = CSDMT_DIR / "examples/makeup"

# 默认推理分辨率
DEFAULT_SIZE = 512

# ========= 工具函数 =========
def np_to_pil(arr: np.ndarray) -> Image.Image:
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L")
    if arr.shape[2] == 4:
        # 如果带 alpha，先转成 RGB（避免下游脚本报错）
        return Image.fromarray(arr[:, :, :3])
    return Image.fromarray(arr)

def save_temp_image(img: Image.Image, suffix=".jpg") -> Path:
    tmpdir = Path(tempfile.mkdtemp(prefix="beauty_web_"))
    path = tmpdir / f"input{suffix}"
    img.save(path, quality=95)
    return path

def run_retouchformer(in_path: Path, out_dir: Path, size: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    # 进入仓库根再调用，和你在 bash 里的做法一致
    cmd = [
        "bash", "-lc",
        f"cd {RETOUCH_DIR} && "
        f"conda run -n {RETOUCH_ENV} python -m retouch_infer.cli "
        f"--input {str(in_path)} "
        f"--output {str(out_dir)} "
        f"--ckpt {str(CKPT_DIR)} "
        f"--epoch {EPOCH} "
        f"--size {size} "
        f"--debug"
    ]
    subprocess.run(cmd, check=True)
    # 优先用原名输出；兜底取最新文件
    candidate = out_dir / in_path.name
    if candidate.exists():
        return candidate
    latest = sorted(out_dir.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not latest:
        raise RuntimeError("RetouchFormer 未产生输出文件")
    return latest[0]

def run_csdmt(content_path: Path, style_path: Path, out_path: Path, size: int, align_skip: bool=True) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    align_arg = "skip" if align_skip else "face"
    cmd = [
        "bash", "-lc",
        f"cd {CSDMT_DIR} && "
        f"conda run -n {CSDMT_ENV} python run_csdmt.py "
        f"--content {str(content_path)} "
        f"--style {str(style_path)} "
        f"--out {str(out_path)} "
        f"--resize {size} "
        f"--align {align_arg}"
    ]
    subprocess.run(cmd, check=True)

# ========= 核心推理函数（返回四张图） =========
def pipeline_gradio(non_makeup_np, makeup_np, size=DEFAULT_SIZE, align_skip=True):
    if non_makeup_np is None or makeup_np is None:
        raise gr.Error("请同时提供【原图】与【美妆参考图】。")

    # 1) 保存上传图为临时文件
    non_makeup_img = np_to_pil(non_makeup_np)
    makeup_img     = np_to_pil(makeup_np)
    tmp_input_path = save_temp_image(non_makeup_img, ".jpg")
    tmp_style_path = save_temp_image(makeup_img, ".jpg")

    # 2) 建临时工作区
    workdir = Path(tempfile.mkdtemp(prefix="beauty_run_"))
    rf_out_dir = workdir / "rf_out"
    final_out  = workdir / "final.jpg"

    # 3) 先修饰（RetouchFormer）
    retouched_path = run_retouchformer(tmp_input_path, rf_out_dir, int(size))

    # 4) 再上妆（CSD-MT）
    run_csdmt(retouched_path, tmp_style_path, final_out, int(size), align_skip=align_skip)

    # 5) 读回四张图：原图 / 参考妆容 / 修饰图 / 成品图
    orig_np   = np.array(non_makeup_img)
    style_np  = np.array(makeup_img)
    ret_np    = np.array(Image.open(retouched_path).convert("RGB"))
    final_np  = np.array(Image.open(final_out).convert("RGB"))

    # 6) 清理输入临时目录（保留工作目录便于调试，可按需清）
    try:
        shutil.rmtree(tmp_input_path.parent, ignore_errors=True)
        shutil.rmtree(tmp_style_path.parent, ignore_errors=True)
    except Exception:
        pass

    return orig_np, style_np, ret_np, final_np

# ========= Gradio UI =========
def build_examples_list(folder: Path):
    if not folder.exists():
        return []
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return sorted([str(p) for p in folder.iterdir() if p.suffix.lower() in exts])

non_makeup_examples = build_examples_list(EX_NON)
makeup_examples     = build_examples_list(EX_MK)

css = """
.gr-button { border-radius: 10px; font-weight: 600; }
#imgbox img { border-radius: 8px; border: 2px solid #eee; }
"""

with gr.Blocks(css=css, theme="soft", title="RetouchFormer + CSD-MT 美颜上妆流水线") as demo:
    gr.Markdown("<h1 style='text-align:center;'>RetouchFormer ➜ CSD-MT 智能美颜上妆</h1>")
    gr.Markdown("先用 RetouchFormer 去瑕疵，再用 CSD-MT 做妆容迁移。一次点击，看到**原图 / 参考妆容 / 修饰图 / 成品图**四图对比。")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 👩 原图（Non-makeup）")
            non_makeup = gr.Image(type="numpy", label="上传未化妆人像", height=320)
            if non_makeup_examples:
                gr.Examples(non_makeup_examples, inputs=[non_makeup], label="示例：未化妆图", examples_per_page=8)

            gr.Markdown("### 💄 参考妆容（Makeup）")
            makeup = gr.Image(type="numpy", label="上传妆容参考图", height=320)
            if makeup_examples:
                gr.Examples(makeup_examples, inputs=[makeup], label="示例：妆容图", examples_per_page=8)

            with gr.Accordion("进阶设置", open=False):
                size = gr.Slider(256, 768, value=DEFAULT_SIZE, step=32, label="推理分辨率（RetouchFormer & CSD-MT 同步）")
                align_skip = gr.Checkbox(value=True, label="CSD-MT 跳过人脸对齐（align=skip）")

            run_btn = gr.Button("✨ 开始处理", variant="primary")

        with gr.Column():
            gr.Markdown("### 结果预览（四图并列）")
            with gr.Row(elem_id="imgbox"):
                out_orig  = gr.Image(label="原图", type="numpy")
                out_style = gr.Image(label="参考妆容", type="numpy")
            with gr.Row(elem_id="imgbox"):
                out_ret   = gr.Image(label="修饰图（RetouchFormer）", type="numpy")
                out_final = gr.Image(label="成品图（CSD-MT）", type="numpy")

    run_btn.click(
        fn=pipeline_gradio,
        inputs=[non_makeup, makeup, size, align_skip],
        outputs=[out_orig, out_style, out_ret, out_final]
    )

# 支持队列，避免并发阻塞；局域网可访问
demo.queue().launch(server_name="0.0.0.0", server_port=7860)
