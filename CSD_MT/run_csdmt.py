# run_csdmt.py
import argparse
import os
import sys
from PIL import Image
from datetime import datetime

HERE = os.path.abspath(os.path.dirname(__file__))

# 让本目录优先进入 sys.path，保证能 import 到本地模块
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from csdmt_api import CSDMTService

def _exists_or_die(p, name):
    if not os.path.exists(p):
        print(f"[FATAL] {name} not found: {p}")
        sys.exit(1)

def main():
    ap = argparse.ArgumentParser(
        description="CSD-MT CLI (no-dlib by default) — verbose"
    )
    ap.add_argument("--content", required=True, help="path to non-makeup image")
    ap.add_argument("--style",   required=True, help="path to makeup reference image")
    ap.add_argument("--out",     required=True, help="path to save output image (e.g. outputs/out.jpg)")
    ap.add_argument("--resize",  type=int, default=256, help="inference resize (256/512)")
    ap.add_argument("--device",  default=None, help="'cuda:0' or 'cpu' (auto if None)")
    ap.add_argument("--align",   default="skip", choices=["skip","dlib"], help="face align mode")
    ap.add_argument("--csdmt_weights",
        default="/root/autodl-tmp/beauty/CSD_MT/CSD_MT/weights/CSD_MT.pth",
        help="path to CSD-MT weights")
    ap.add_argument("--fp_weights",
        default="/root/autodl-tmp/beauty/CSD_MT/faceutils/face_parsing/res/cp/79999_iter.pth",
        help="path to face parsing (BiSeNet) weights")
    args = ap.parse_args()

    print("========== CSD-MT CLI ==========")
    print("[Run] cwd:", os.getcwd())
    print("[Run] script dir:", HERE)
    print("[Args] content:", args.content)
    print("[Args] style  :", args.style)
    print("[Args] out    :", args.out)
    print("[Args] resize :", args.resize)
    print("[Args] device :", args.device)
    print("[Args] align  :", args.align)
    print("[Args] csdmt_weights:", args.csdmt_weights)
    print("[Args] fp_weights   :", args.fp_weights)

    # 路径存在性检查
    _exists_or_die(args.content, "content image")
    _exists_or_die(args.style,   "style image")
    _exists_or_die(args.csdmt_weights, "CSD-MT weights")
    _exists_or_die(args.fp_weights,    "Face Parsing weights")

    # 创建输出目录
    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)

    # 加载服务（这里内部会打印 device / 权重路径）
    print("[Stage] init CSDMTService ...")
    svc = CSDMTService(
        csdmt_weights=args.csdmt_weights,
        faceparsing_weights=args.fp_weights,
        device=args.device,
        align_mode=args.align,
        resize=args.resize
    )

    # 读图
    print("[Stage] load images ...")
    content = Image.open(args.content).convert("RGB")
    style   = Image.open(args.style).convert("RGB")
    print("[Info] content size:", content.size, "| style size:", style.size)

    # 推理
    print("[Stage] inference ...")
    start_ts = datetime.now()
    result = svc.transfer(content, style)
    dt = (datetime.now() - start_ts).total_seconds()
    print(f"[Perf] inference time: {dt:.3f}s")

    # 保存
    Image.fromarray(result).save(args.out)
    print(f"[OK] saved: {args.out}")
    print("=========== DONE ===========")

if __name__ == "__main__":
    main()
