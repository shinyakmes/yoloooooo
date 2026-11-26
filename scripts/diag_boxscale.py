# -*- coding: utf-8 -*-
"""检验「模型看到了缺陷、但框的形状/尺度不对」这个假设 —— 用数字，不用感觉。

diag_class.py 回答的是「找得到但框不准？」（AP@0.1 高、AP@0.5 低）。
这个脚本回答下一层：**框具体错在哪、错在哪个置信度档上？**

方法（= 真实部署的行为，不是上界）：
  1. 每张图、每一类，把预测按置信度从高到低排序；
  2. 逐框贪心认领还没被认领的同类 GT（IoU >= --match-iou 才算认领）；
  3. 对每个认领成功的 (pred, GT) 对，记录 conf / IoU / 面积比 / 中心偏移；
  4. 按置信度分档统计。

判读（写在输出里，不用记）：
  - 高置信度档的框质量好、低档差  -> 是「低对比导致置信度上不去」的信号问题，不是框的问题
  - 各档面积比中位都 << 1        -> 模型只框住局部子区域
  - 各档中心偏移都大              -> 定位漂移
  - 未认领比例高                  -> 模型真的没看见

⚠️ 反面教材（我第一版就是这么写错的）：对每个 GT 在 300 个候选里挑"最像的那个"。
   门限 0.001 时模型有无限量猜测空间，必然猜中，于是得到"框大小完全正确"的假结论。
   评价必须用"模型主动报出来的框"，不能用上界。

用法：
    python diag_boxscale.py --weights E:\\yolo\\runs\\baseline_640\\weights\\best.pt --tag v8n50
    python diag_boxscale.py --weights <best.pt> --tag clahe --data-root E:\\yolo\\datasets\\neu-det-yolo-clahe
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from ultralytics import YOLO

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE = os.environ.get('YOLO_ROOT', r'E:\yolo')
OUT = os.path.join(BASE, 'runs', 'eval')
CLASSES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
BINS = [(0.0, 0.10, '<0.10'), (0.10, 0.25, '0.10-0.25'), (0.25, 1.01, '>=0.25')]


def iou_mat(a, b):
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
                         (cx + bw / 2) * w, (cy + bh / 2) * h])
    return np.array(rows, dtype=np.float32).reshape(-1, 5)


def med(a):
    return float(np.median(a)) if len(a) else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--split', default='test')
    ap.add_argument('--data-root', default=os.path.join(BASE, 'datasets', 'neu-det-yolo'))
    ap.add_argument('--pred-conf', type=float, default=0.001)
    ap.add_argument('--match-iou', type=float, default=0.10,
                    help='认领门限；放低一点，好让"报了但框歪"的也能被统计到')
    args = ap.parse_args()

    imgs_dir = os.path.join(args.data_root, 'images', args.split)
    lbls_dir = os.path.join(args.data_root, 'labels', args.split)
    files = sorted(glob.glob(os.path.join(imgs_dir, '*.jpg')) +
                   glob.glob(os.path.join(imgs_dir, '*.png')))
    if not files:
        raise SystemExit('no images under %s' % imgs_dir)
    model = YOLO(args.weights)

    n_gt = {c: 0 for c in CLASSES}
    n_claimed = {c: 0 for c in CLASSES}
    n_claimed_hi = {c: 0 for c in CLASSES}          # conf >= 0.25 认领到的
    pairs = {c: [] for c in CLASSES}                # (conf, iou, area_ratio, offset)

    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        r = model.predict(f, imgsz=args.imgsz, conf=args.pred_conf, iou=0.6,
                          verbose=False, max_det=300)[0]
        h, w = r.orig_shape
        pc = r.boxes.cls.cpu().numpy().astype(int) if len(r.boxes) else np.zeros(0, int)
        pb = r.boxes.xyxy.cpu().numpy() if len(r.boxes) else np.zeros((0, 4), np.float32)
        ps = r.boxes.conf.cpu().numpy() if len(r.boxes) else np.zeros(0, np.float32)
        gt = load_gt(os.path.join(lbls_dir, stem + '.txt'), w, h)
        if len(gt) == 0:
            continue

        for ci, cname in enumerate(CLASSES):
            gi = np.where(gt[:, 0].astype(int) == ci)[0]
            if not len(gi):
                continue
            n_gt[cname] += len(gi)
            pi = np.where(pc == ci)[0]
            if not len(pi):
                continue
            gb = gt[gi, 1:5]
            order = pi[np.argsort(-ps[pi])]
            M = iou_mat(pb[order], gb)
            matched = np.zeros(len(gi), dtype=bool)
            for k, idx in enumerate(order):
                cand = np.where(~matched)[0]
                if not len(cand):
                    break
                j = int(cand[int(np.argmax(M[k, cand]))])
                iou = float(M[k, j])
                if iou < args.match_iou:
                    continue
                matched[j] = True
                conf = float(ps[idx])
                pred, g = pb[idx], gb[j]
                pa = max((pred[2] - pred[0]) * (pred[3] - pred[1]), 1e-9)
                ga = max((g[2] - g[0]) * (g[3] - g[1]), 1e-9)
                gw = max(g[2] - g[0], 1e-9)
                gh = max(g[3] - g[1], 1e-9)
                dcx = ((pred[0] + pred[2]) / 2 - (g[0] + g[2]) / 2) / gw
                dcy = ((pred[1] + pred[3]) / 2 - (g[1] + g[3]) / 2) / gh
                pairs[cname].append((conf, iou, pa / ga, float(np.hypot(dcx, dcy))))
                n_claimed[cname] += 1
                if conf >= 0.25:
                    n_claimed_hi[cname] += 1

    # ---------- 1. GT 被认领情况 ----------
    print('\n=== 1. GT 被认领情况（贪心按置信度认领，认领门限 IoU>=%.2f）===' % args.match_iou)
    print('  %-16s %6s %14s %14s %8s' % ('class', 'n_gt', '被认领(全门限)', '被认领(conf>=.25)', '未认领'))
    for c in CLASSES:
        if not n_gt[c]:
            continue
        print('  %-16s %6d %8d (%4.0f%%) %8d (%4.0f%%) %8d'
              % (c, n_gt[c], n_claimed[c], 100.0 * n_claimed[c] / n_gt[c],
                 n_claimed_hi[c], 100.0 * n_claimed_hi[c] / n_gt[c],
                 n_gt[c] - n_claimed[c]))

    # ---------- 2. 认领框的质量 × 置信度分档 ----------
    print('\n=== 2. 认领框的质量 × 置信度分档 ===')
    print('  %-16s %-11s %5s %8s %8s %10s %8s %9s' %
          ('class', 'conf档', 'n', 'IoU中位', '面比中位', '<0.5倍占比', '中心偏', '面比p25'))
    table = {}
    for c in CLASSES:
        if not pairs[c]:
            continue
        arr = np.array(pairs[c], dtype=np.float32)
        table[c] = {}
        for lo, hi, label in BINS:
            sel = arr[(arr[:, 0] >= lo) & (arr[:, 0] < hi)]
            if not len(sel):
                continue
            small = 100.0 * float((sel[:, 2] < 0.5).mean())
            table[c][label] = {
                'n': int(len(sel)), 'iou_median': med(sel[:, 1]),
                'area_ratio_median': med(sel[:, 2]),
                'frac_lt_half': float((sel[:, 2] < 0.5).mean()),
                'center_offset_median': med(sel[:, 3]),
            }
            print('  %-16s %-11s %5d %8.3f %8.2f %9.1f%% %8.3f %9.2f'
                  % (c, label, len(sel), med(sel[:, 1]), med(sel[:, 2]),
                     small, med(sel[:, 3]), float(np.percentile(sel[:, 2], 25))))

    # ---------- 3. 判读 ----------
    print('\n=== 3. 判读 ===')
    for c in CLASSES:
        if c not in table or '>=0.25' not in table[c]:
            continue
        hi = table[c]['>=0.25']
        claim = 100.0 * n_claimed_hi[c] / n_gt[c]
        print('  %-16s conf>=0.25 认领率 %5.1f%% | 该档面比中位 %.2f | IoU中位 %.3f | 中心偏 %.3f'
              % (c, claim, hi['area_ratio_median'], hi['iou_median'],
                 hi['center_offset_median']))
    print('\n  读法：')
    print('   - 认领率高但 conf>=0.25 认领率低 -> 模型看得见，但置信度上不去（低对比的信号问题）')
    print('   - 面比中位 <0.6 -> 只框住局部子区域；>1.6 -> 框进了背景')
    print('   - 中心偏 >0.2 -> 定位漂移')

    out = {
        'weights': args.weights, 'imgsz': args.imgsz, 'split': args.split,
        'data_root': args.data_root, 'match_iou': args.match_iou,
        'n_gt': n_gt, 'n_claimed': n_claimed, 'n_claimed_conf_ge_025': n_claimed_hi,
        'quality_by_conf_bin': table,
    }
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, 'boxscale_%s.json' % args.tag)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print('\n  saved -> %s' % path)


if __name__ == '__main__':
    main()
