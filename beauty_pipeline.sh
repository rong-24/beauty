#!/usr/bin/env bash
set -euo pipefail

# ===== 配置：按你的实际路径/环境名改 =====
CSDMT_ENV="CSDMT"                       # 你的 CSDMT conda 环境名
RETOUCH_ENV="Retouchformer"             # 你的 Retouchformer conda 环境名

CSDMT_DIR="/root/autodl-tmp/beauty/CSD_MT"
RETOUCH_DIR="/root/autodl-tmp/beauty/retouchformer"

CKPT_DIR="${RETOUCH_DIR}/release_model" # RetouchFormer 模型目录（绝对路径，稳）
EPOCH="best"                             # RetouchFormer ckpt 标记

# GPU 选择（如需切换 GPU，改成相应的编号；CPU 跑就注释掉）
export CUDA_VISIBLE_DEVICES=0

# ===== 用法提示 =====
usage() {
  cat <<'USAGE'
用法：
  单张：
    ./beauty_pipeline.sh --content /path/to/a.jpg --style /path/to/style.jpg --out /path/to/out.jpg [--size 512]

  批量（对 content_dir 中的每张图，统一使用同一 style）：
    ./beauty_pipeline.sh --content_dir /path/to/dir --style /path/to/style.jpg --out_dir /path/to/out_dir [--size 512]

参数：
  --content      单张待处理人像
  --style        参考妆容图
  --out          单张输出路径（会自动建目录）
  --content_dir  批量输入目录（jpg/png/jpeg/webp）
  --out_dir      批量输出目录
  --size         RetouchFormer 推理分辨率（默认 512），与 CSD-MT 的 --resize 同步
USAGE
}

# ===== 解析参数 =====
CONTENT=""
STYLE=""
OUT=""
CONTENT_DIR=""
OUT_DIR=""
SIZE=512

while [[ $# -gt 0 ]]; do
  case "$1" in
    --content) CONTENT="$2"; shift 2 ;;
    --style) STYLE="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --content_dir) CONTENT_DIR="$2"; shift 2 ;;
    --out_dir) OUT_DIR="$2"; shift 2 ;;
    --size) SIZE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "未知参数：$1"; usage; exit 1 ;;
  esac
done

if [[ -z "$STYLE" ]]; then echo "缺少 --style"; usage; exit 1; fi

# 临时目录 & 清理
WORKDIR="$(mktemp -d -t beauty_pipe_XXXXXX)"
trap 'rm -rf "$WORKDIR"' EXIT

run_one() {
  local in_img="$1"
  local out_img="$2"
  local basename="$(basename "$in_img")"
  local rf_out_dir="$WORKDIR/rf_out"
  mkdir -p "$rf_out_dir"
  mkdir -p "$(dirname "$out_img")"

  echo "[1/2] RetouchFormer 修饰：$in_img -> $rf_out_dir/"
  # 关键改动：用子 shell 先 cd 到仓库根，再 conda run 直呼 python
  (
    cd "$RETOUCH_DIR"
    conda run -n "$RETOUCH_ENV" python -m retouch_infer.cli \
      --input "$in_img" \
      --output "$rf_out_dir" \
      --ckpt "$CKPT_DIR" \
      --epoch "$EPOCH" \
      --size "$SIZE" \
      --debug
  )

  # RetouchFormer 会按原文件名写出到 output 目录
  local retouched="${rf_out_dir}/${basename}"
  if [[ ! -f "$retouched" ]]; then
    # 兜底：取最新生成的文件
    retouched="$(ls -1t "$rf_out_dir" | head -n1)"
    retouched="$rf_out_dir/$retouched"
  fi

  echo "[2/2] CSD-MT 上妆：$retouched + $STYLE -> $out_img"
  (
    cd "$CSDMT_DIR"
    conda run -n "$CSDMT_ENV" python run_csdmt.py \
      --content "$retouched" \
      --style "$STYLE" \
      --out "$out_img" \
      --resize "$SIZE" \
      --align skip
  )

  echo "✅ 完成：$out_img"
}

# 单张模式
if [[ -n "$CONTENT" && -n "$OUT" ]]; then
  run_one "$CONTENT" "$OUT"
  exit 0
fi

# 批量模式
if [[ -n "$CONTENT_DIR" && -n "$OUT_DIR" ]]; then
  shopt -s nullglob nocaseglob
  for f in "$CONTENT_DIR"/*.{jpg,jpeg,png,webp}; do
    rel="$(basename "$f")"
    run_one "$f" "$OUT_DIR/$rel"
  done
  echo "🎉 批量完成，输出在：$OUT_DIR"
  exit 0
fi

echo "参数不完整。"
usage
exit 1
