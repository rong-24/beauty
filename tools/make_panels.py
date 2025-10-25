#!/usr/bin/env python3
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

root = Path("/root/autodl-tmp/beauty/outputs/outputs_batch_bold")
save_dir = root / "panels"
save_dir.mkdir(exist_ok=True)

# 遍历每个子文件夹
for subdir in sorted(root.iterdir()):
    if not subdir.is_dir():
        continue

    # 查找四张关键图
    img_paths = {
        "原图": subdir / "content.jpg",
        "参考妆": subdir / "style.jpg",
        "初步妆": subdir / "csdmt.jpg",
        "优化后": subdir / "optimized.jpg"
    }
    if not all(p.exists() for p in img_paths.values()):
        print(f"[跳过] {subdir.name}: 缺少图片")
        continue

    # 打开 & 统一高度
    imgs = [Image.open(p).convert("RGB") for p in img_paths.values()]
    target_h = 512
    imgs = [im.resize((int(im.width * target_h / im.height), target_h)) for im in imgs]

    # 字体
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 24)
    except Exception:
        font = ImageFont.load_default()

    labeled = []
    for (label, im) in zip(img_paths.keys(), imgs):
        draw = ImageDraw.Draw(im)
        w, h = im.size

        # ✅ Pillow 10 及以上用 textbbox 获取文字尺寸
        try:
            bbox = draw.textbbox((0, 0), label, font=font)
            text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            # 兼容旧版本 Pillow
            text_w, text_h = draw.textsize(label, font=font)

        # 底部黑条+文字
        draw.rectangle([0, h - text_h - 10, text_w + 10, h], fill=(0, 0, 0))
        draw.text((5, h - text_h - 5), label, fill=(255, 255, 255), font=font)
        labeled.append(im)

    # 横向拼接
    widths = [im.width for im in labeled]
    total_w = sum(widths)
    panel = Image.new("RGB", (total_w, target_h))
    x = 0
    for im in labeled:
        panel.paste(im, (x, 0))
        x += im.width

    out_path = save_dir / f"{subdir.name}_panel.jpg"
    panel.save(out_path, quality=95)
    print(f"[保存] {out_path}")

print(f"\n✅ 所有拼接完成，已保存到: {save_dir}")
