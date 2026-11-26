# -*- coding: utf-8 -*-
"""Diagnose WHY one class is weak: is it "cannot see it" or "cannot localize it"?

Two measurements, both on the same test split:

1. GT box geometry per class: how much of the image does a typical GT box cover?
   A class whose boxes are 60-100% of the tile is a whole-field texture defect,
   and bounding-box regression is a poor task formulation for it.

2. Per-class AP at several IoU thresholds (own greedy matcher, 101-point
   interpolation). Reading of the sweep:
     AP@0.1 high + AP@0.5 low  -> the model finds the defect but the box is
                                  imprecise; a box is the wrong output format.
     AP@0.1 also low           -> the model genuinely does not see it.

The AP@0.5 column is cross-checked against `yolo val` (must agree within ~0.01),
so this file's numbers are safe to quote.

Usage:
    python diag_class.py --weights <best.pt> [--cls crazing] [--imgsz 640]
"""
import argparse
import glob
import json
import os

import numpy as np
from ultralytics import YOLO

BASE = os.environ.get('YOLO_ROOT', r'E:\yolo')
DATA = os.path.join(BASE, 'datasets', 'neu-det-yolo')
LABELS = os.path.join(DATA, 'labels', 'test')
IMAGES = os.path.join(DATA, 'images', 'test')
OUT = os.path.join(BASE, 'runs', 'eval')
CLASSES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']


def load_gt(label_path, w, h):
    rows = []
    if os.path.exists(label_path):
        for line in open(label_path, encoding='utf-8'):
            p = line.split()
            if len(p) < 5:
                continue
            c = int(float(p[0]))
            cx, cy, bw, bh = (float(v) for v in p[1:5])
            rows.append([c, (cx - bw / 2) * w, (cy - bh / 2) * h,
                         (cx + bw / 2) * w, (cy + bh / 2) * h, bw * bh])
    return np.array(rows, dtype=np.float32).reshape(-1, 6)


def iou_mat(a, b):
    """(N,4) x (M,4) -> (N,M) IoU"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    ab = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-9)


def ap_101(confs, tps, n_gt):
    """All-point (101-step) AP from a score-sorted list of TP flags."""
    if n_gt == 0 or len(confs) == 0:
        return 0.0
    order = np.argsort(-confs)
    tps = tps[order]
    ctp = np.cumsum(tps)
    cfp = np.cumsum(1 - tps)
    rec = ctp / n_gt
    prec = ctp / np.maximum(ctp + cfp, 1e-9)
    grid = np.linspace(0, 1, 101)
    p = np.interp(grid, rec, prec)
    # make precision monotonically non-increasing to the right (standard envelope)
    for i in range(len(p) - 2, -1, -1):
        p[i] = max(p[i], p[i + 1])
    return float(p.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--tag', default='diag')
    ap.add_argument('--cls', default='crazing')
    ap.add_argument('--pred-conf', type=float, default=0.001,
                    help='predict floor; keep it at 0.001 to match yolo val')
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob(os.path.join(IMAGES, '*.jpg')))
    model = YOLO(args.weights)

    # ---------- 1. GT geometry ----------
    area_frac = {c: [] for c in CLASSES}
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        gt = load_gt(os.path.join(LABELS, stem + '.txt'), 200, 200)
        for row in gt:
            area_frac[CLASSES[int(row[0])]].append(float(row[5]))

    print('=== 1. 真框尺度 (框面积 / 整图面积, 图像 200x200) ===')
    print('  %-16s %5s %8s %8s %8s' % ('class', 'n', 'median', 'p90', 'max'))
    for c in CLASSES:
        a = np.array(area_frac[c]) if area_frac[c] else np.array([0.0])
        print('  %-16s %5d %7.1f%% %7.1f%% %7.1f%%'
              % (c, len(a), 100 * np.median(a), 100 * np.percentile(a, 90), 100 * a.max()))

    # ---------- 2. collect predictions once ----------
    preds, gts = [], []
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        r = model.predict(f, imgsz=args.imgsz, conf=args.pred_conf, iou=0.6,
                          verbose=False, max_det=300)[0]
        h, w = r.orig_shape
        if len(r.boxes):
            preds.append((r.boxes.cls.cpu().numpy().astype(int),
                          r.boxes.conf.cpu().numpy(),
                          r.boxes.xyxy.cpu().numpy()))
        else:
            preds.append((np.zeros(0, dtype=int), np.zeros(0, np.float32), np.zeros((0, 4), np.float32)))
        gt = load_gt(os.path.join(LABELS, stem + '.txt'), w, h)
        gts.append((gt[:, 0].astype(int) if len(gt) else np.zeros(0, dtype=int),
                    gt[:, 1:5].astype(np.float32) if len(gt) else np.zeros((0, 4), np.float32)))

    n_gt = {c: int(sum((g[0] == ci).sum() for g in gts)) for ci, c in enumerate(CLASSES)}

    # ---------- 3. AP vs IoU threshold ----------
    ious = [0.1, 0.3, 0.5, 0.7]
    print('\n=== 2. 各类 AP 在不同 IoU 门限下 (同一批预测, 同一测试集) ===')
    print('  %-16s %8s %8s %8s %8s' % ('class', 'IoU.1', 'IoU.3', 'IoU.5', 'IoU.7'))
    table = {}
    for ci, cname in enumerate(CLASSES):
        row = {}
        for thr in ious:
            confs, tps = [], []
            for (pc, ps, pb), (gc, gb) in zip(preds, gts):
                pi = np.where(pc == ci)[0]
                gi = np.where(gc == ci)[0]
                if len(pi) == 0:
                    continue
                order = pi[np.argsort(-ps[pi])]
                matched = np.zeros(len(gi), dtype=bool)
                if len(gi):
                    M = iou_mat(pb[order], gb[gi])
                for k, pi_k in enumerate(order):
                    hit = False
                    if len(gi):
                        cand = np.where(~matched)[0]
                        if len(cand):
                            j = cand[int(np.argmax(M[k, cand]))]
                            if M[k, j] >= thr:
                                matched[j] = True
                                hit = True
                    confs.append(ps[pi_k])
                    tps.append(1.0 if hit else 0.0)
            row[thr] = ap_101(np.array(confs), np.array(tps), n_gt[cname])
        table[cname] = row
        print('  %-16s %7.3f %8.3f %8.3f %8.3f'
              % (cname, row[0.1], row[0.3], row[0.5], row[0.7]))

    allap = {t: float(np.mean([table[c][t] for c in CLASSES])) for t in ious}
    print('  %-16s %7.3f %8.3f %8.3f %8.3f' % ('(mAP)', allap[0.1], allap[0.3], allap[0.5], allap[0.7]))

    # ---------- 4. where does the focus class score sit ----------
    focus = args.cls
    for cname in (focus,):
        ci = CLASSES.index(cname)
        tp_scores, fp_scores = [], []
        per_img = []  # (conf, is_tp) for threshold sweep
        for (pc, ps, pb), (gc, gb) in zip(preds, gts):
            pi = np.where(pc == ci)[0]
            gi = np.where(gc == ci)[0]
            if len(pi) == 0:
                continue
            order = pi[np.argsort(-ps[pi])]
            matched = np.zeros(len(gi), dtype=bool)
            M = iou_mat(pb[order], gb[gi]) if len(gi) else np.zeros((len(order), 0))
            for k, pi_k in enumerate(order):
                hit = False
                if len(gi):
                    cand = np.where(~matched)[0]
                    if len(cand):
                        j = cand[int(np.argmax(M[k, cand]))]
                        if M[k, j] >= 0.5:
                            matched[j] = True
                            hit = True
                (tp_scores if hit else fp_scores).append(float(ps[pi_k]))
                per_img.append((float(ps[pi_k]), hit))
        tp_scores, fp_scores = np.array(tp_scores), np.array(fp_scores)
        print('\n=== 3. %s 的置信度分布 (IoU>=0.5 判定) ===' % cname)
        if len(tp_scores):
            print('  报对的框 %3d 个, 置信度中位数 %.3f, 四分位 %.3f / %.3f'
                  % (len(tp_scores), np.median(tp_scores),
                     np.percentile(tp_scores, 25), np.percentile(tp_scores, 75)))
        if len(fp_scores):
            print('  虚报的框 %3d 个, 置信度中位数 %.3f, 四分位 %.3f / %.3f'
                  % (len(fp_scores), np.median(fp_scores),
                     np.percentile(fp_scores, 25), np.percentile(fp_scores, 75)))

        # ---------- 5. what a per-class threshold would buy ----------
        n_true = n_gt[cname]
        print('\n=== 4. 单类门限扫描 %s (IoU>=0.5): 把这一类的门限单独调低会怎样 ===' % cname)
        print('  %6s %6s %6s %8s %8s %8s' % ('conf', 'TP', 'FP', 'precision', 'recall', 'F1'))
        sweep = []
        for thr in (0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50):
            tp = sum(1 for c, h in per_img if h and c >= thr)
            fp = sum(1 for c, h in per_img if (not h) and c >= thr)
            p = tp / (tp + fp) if tp + fp else 0.0
            r = tp / n_true if n_true else 0.0
            f1 = 2 * p * r / (p + r) if p + r else 0.0
            sweep.append({'conf': thr, 'tp': tp, 'fp': fp, 'p': p, 'r': r, 'f1': f1})
            print('  %6.2f %6d %6d %7.1f%% %7.1f%% %7.3f'
                  % (thr, tp, fp, 100 * p, 100 * r, f1))

    out = {'weights': args.weights, 'imgsz': args.imgsz, 'n_gt': n_gt,
           'ap_vs_iou': {c: {str(k): v for k, v in t.items()} for c, t in table.items()},
           'mAP_vs_iou': {str(k): v for k, v in allap.items()},
           'threshold_sweep_%s' % focus: sweep}
    path = os.path.join(OUT, 'diag_%s.json' % args.tag)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print('\n  saved -> %s' % path)


if __name__ == '__main__':
    main()
