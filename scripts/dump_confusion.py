# -*- coding: utf-8 -*-
"""Dump the numeric confusion matrix of a trained run.

Orientation follows ultralytics: matrix[predicted, true], so
    - cell [i, j]        = 真实是 j 类，却被预测成 i 类的框数
    - 对角线 [i, i]       = 判对的框数
    - 最后一行             = 真实存在但模型什么都没预测到（漏检 FN）
    - 最后一列             = 模型预测了但附近没有对应的真框（误检 FP）

Splitting by column (per true class) reproduces confusion_matrix_normalized.png.

Usage:
    python dump_confusion.py <weights> [--split test] [--conf 0.25] [--iou 0.6]
"""
import argparse
import os
from pathlib import Path

import numpy as np
from ultralytics import YOLO

# data.yaml lives outside the git repo by default; override with YOLO_ROOT.
BASE = Path(os.environ.get("YOLO_ROOT", r"E:\yolo"))
DATA = BASE / "datasets" / "neu-det-yolo" / "data.yaml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("--split", default="test")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.6)
    args = ap.parse_args()

    model = YOLO(args.weights)
    # ultralytics only fills the confusion matrix when plots=True; send the
    # regenerated plots to a scratch dir so the original run folder stays intact.
    scratch = BASE / "runs" / "_scratch_val"
    m = model.val(data=str(DATA), split=args.split, conf=args.conf, iou=args.iou,
                  plots=True, verbose=False, project=str(scratch), name="cm",
                  exist_ok=True)

    names = [m.names[i] for i in range(len(m.names))]
    cm = m.confusion_matrix.matrix  # shape (nc + 1, nc + 1) = [pred, true]
    labels = names + ["无框"]
    short = [n[:9] for n in names] + ["无框"]

    col_tot = cm.sum(axis=0)
    tag = Path(args.weights).parts

    print(f"\n=== {'/'.join(tag[-4:-2])}  split={args.split}  conf={args.conf} ===")
    print("列 = 真实类别, 行 = 模型预测; 每列归一化到 100% (= 该类的框都去了哪)\n")
    print("预测\\真实".ljust(12) + "".join(f"{s:>11}" for s in short))
    for i, s in enumerate(short):
        cells = "".join(
            f"{100.0 * cm[i, j] / col_tot[j]:>10.1f}%" if col_tot[j] > 0 else f"{'-':>11}"
            for j in range(len(short))
        )
        print(s.ljust(12) + cells)

    print("\n原始框数 (列 = 真实, 行 = 预测):")
    print("预测\\真实".ljust(12) + "".join(f"{s:>11}" for s in short))
    for i, s in enumerate(short):
        print(s.ljust(12) + "".join(f"{int(cm[i, j]):>11}" for j in range(len(short))))

    print("\n各类召回 (对角线 / 该类真实框数) 与误检占比:")
    for c, name in enumerate(names):
        if col_tot[c] == 0:
            continue
        rec = cm[c, c] / col_tot[c]
        miss = cm[len(names), c] / col_tot[c]
        print(f"  {name:<15} 召回 {rec:6.1%}  漏检 {miss:6.1%}  (真框 {int(col_tot[c])})")

    bg_col = cm[:, len(names)]
    print(f"\n总误检框 {int(bg_col.sum())} 个 (预测了但没有真框匹配), 按预测类别:")
    for c, name in enumerate(names):
        if bg_col[c]:
            print(f"  {name:<15} {int(bg_col[c])}")
    print(f"真框总数 {int(col_tot[:len(names)].sum())}")


if __name__ == "__main__":
    main()
