#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, csv, re
import numpy as np

try:
    import pandas as pd
except Exception:
    print("[ERROR] 需要 pandas：pip install pandas")
    raise

try:
    from scipy.stats import spearmanr, kendalltau
except Exception:
    print("[ERROR] 需要 scipy：pip install scipy")
    raise


# ---------- 工具：尝试多编码 + 自动分隔符 ----------
def _read_csv_robust(path: str) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "gbk", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            # 自动识别分隔符
            with open(path, "rb") as fb:
                sample = fb.read(4096)
            try:
                sample_txt = sample.decode(enc, errors="strict")
            except Exception as e:
                last_err = e
                continue
            try:
                dialect = csv.Sniffer().sniff(sample_txt, delimiters=",;\t|")
                sep = dialect.delimiter
            except Exception:
                # 默认逗号
                sep = ","
            df = pd.read_csv(path, encoding=enc, sep=sep)
            print(f"[Info] 读取成功：encoding={enc}, sep='{sep}'")
            return df
        except Exception as e:
            last_err = e
            continue
    raise last_err or RuntimeError("CSV 解码失败")

# ---------- 清洗到数值 ----------
def _coerce_num(s: pd.Series) -> pd.Series:
    if s.dtype.kind in "if":
        return s.astype(float)
    # 去掉百分号、空格、千分位等
    s = s.astype(str).str.replace(r"[,%\s]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")

# ---------- 加载数据表 ----------
def load_table(csv_path: str | None, mos_col: str | None = None) -> pd.DataFrame:
    if csv_path:
        df = _read_csv_robust(csv_path)
    else:
        # 演示用内置数据
        data = {
            "IDSim":   [0.7975, 0.7103, 0.7696, 0.7700, 0.8477, 0.7307, 0.8253],
            "StyleSim":[0.9979, 0.9806, 0.9970, 0.9978, 0.9980, 0.9813, 0.9966],
            "SSIM":    [0.8912, 0.8648, 0.8811, 0.8824, 0.9143, 0.9021, 0.9119],
            "MOS":     [0.96,   0.82,   0.85,   0.91,   0.73,   0.80,   0.80],
        }
        df = pd.DataFrame(data)
        print("[Info] 未提供 CSV，使用内置示例数据。")

    # 列名标准化（去空白）
    df.columns = [str(c).strip() for c in df.columns]

    # 选择 MOS 列：优先用户指定；否则自动找 MOS / MOS_norm / Normalized_MOS
    mos_candidates = [mos_col] if mos_col else ["MOS", "MOS_norm", "Normalized_MOS", "MOS(0-1)"]
    mos_col_real = None
    for c in mos_candidates:
        if c and c in df.columns:
            mos_col_real = c
            break
    if mos_col_real is None:
        raise ValueError(f"未找到 MOS 列；请在 CSV 中包含 'MOS' 或 'MOS_norm' 等，或用 --mos-col 指定。现有列：{list(df.columns)}")

    need = {"IDSim", "StyleSim", "SSIM", mos_col_real}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"缺少列：{missing}；现有列：{list(df.columns)}")

    # 提取并转为数值
    use = df[list(need)].copy()
    use.rename(columns={mos_col_real: "MOS"}, inplace=True)
    for col in ["IDSim", "StyleSim", "SSIM", "MOS"]:
        use[col] = _coerce_num(use[col])

    # 丢弃无效行
    before = len(use)
    use = use.dropna(subset=["IDSim", "StyleSim", "SSIM", "MOS"])
    after = len(use)
    if after < before:
        print(f"[Warn] 有 {before-after} 行包含非数字/缺失，已丢弃。")

    # 归一化 MOS 与否都不影响 Spearman；这里仅提示
    if (use["MOS"] < 0).any() or (use["MOS"] > 1).any():
        print("[Info] 检测到 MOS 不是 0-1 区间，这不影响 Spearman/Kendall。")

    return use.reset_index(drop=True)


# ---------- 网格搜索 ----------
def grid_search_lambdas(df: "pd.DataFrame", step: float = 0.05, topk: int = 5):
    ID = df["IDSim"].to_numpy(float)
    ST = df["StyleSim"].to_numpy(float)
    SS = df["SSIM"].to_numpy(float)
    MOS = df["MOS"].to_numpy(float)

    best = {"rho": -2, "tau": -2, "lam": None, "acs": None}
    records = []
    grid = np.arange(0.0, 1.0 + 1e-12, step)

    for l1 in grid:
        for l2 in grid:
            l3 = 1.0 - l1 - l2
            if l3 < 0:  # 不满足简单约束
                continue
            l = np.array([l1, l2, l3], dtype=float)
            s = l.sum()
            if s <= 0: 
                continue
            l /= s  # 归一到和=1
            l1_, l2_, l3_ = l.tolist()

            ACS = l1_*ID + l2_*ST + l3_*SS
            rho, _ = spearmanr(ACS, MOS)
            tau, _ = kendalltau(ACS, MOS)

            rec = {"rho": float(rho), "tau": float(tau), "lam": (l1_, l2_, l3_)}
            records.append(rec)
            if rho > best["rho"]:
                best.update({"rho": float(rho), "tau": float(tau), "lam": (l1_, l2_, l3_), "acs": ACS})

    records_sorted = sorted(records, key=lambda r: r["rho"], reverse=True)[:topk]
    return best, records_sorted


def main():
    import argparse
    ap = argparse.ArgumentParser(description="在一张 CSV 上网格搜索 λ，使 Spearman(ACS, MOS) 最大")
    ap.add_argument("csv", nargs="?", default=None, help="CSV 路径（必须包含列：IDSim, StyleSim, SSIM, MOS或MOS_norm）")
    ap.add_argument("--mos-col", default=None, help="如果 MOS 列名不是 MOS（如 MOS_norm），用此参数指定")
    ap.add_argument("--step", type=float, default=float(os.environ.get("LAMBDA_STEP", 0.05)), help="网格步长（越小越精细，越慢）")
    ap.add_argument("--topk", type=int, default=5, help="打印前 K 组最优 λ")
    args = ap.parse_args()

    df = load_table(args.csv, mos_col=args.mos_col)
    best, topk = grid_search_lambdas(df, step=args.step, topk=args.topk)

    print("\n========== 结果 ==========")
    l1, l2, l3 = best["lam"]
    print(f"最优 λ: λ1(ID)={l1:.3f}, λ2(Style)={l2:.3f}, λ3(Struct)={l3:.3f}  (和={l1+l2+l3:.3f})")
    print(f"Spearman 相关(ACS vs MOS): {best['rho']:.4f}")
    print(f"Kendall  相关(ACS vs MOS): {best['tau']:.4f}")

    print("\nTop-{} 组合（按 Spearman 排序）：".format(args.topk))
    for i, rec in enumerate(topk, 1):
        a, b, c = rec["lam"]
        print(f"{i:>2d}) ρ={rec['rho']:.4f}, τ={rec['tau']:.4f} | λ=({a:.3f}, {b:.3f}, {c:.3f})")

    # 输出带最佳 ACS 的 CSV（便于后续画图）
    out = df.copy()
    out["ACS_best"] = best["acs"]
    if args.csv:
        stem, ext = os.path.splitext(args.csv)
        out_path = f"{stem}_with_best.csv"
    else:
        out_path = "acs_with_best.csv"
    out.to_csv(out_path, index=False)
    print(f"\n[Saved] 已输出带最佳 ACS 的表：{out_path}")


if __name__ == "__main__":
    main()
