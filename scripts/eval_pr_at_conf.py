# -*- coding: utf-8 -*-
"""
Precision / recall / miss rate at an EXPLICIT confidence threshold, with our own
greedy matcher.

Why not trust `res.box.mp / res.box.mr` from `model.val()`: those are read off the
PR curve at the max-F1 point, so they barely move when the `conf` argument
changes (a sweep over conf=0.05..0.20 returned byte-identical numbers, which is
what exposed it). For a report/resume number we want to control the threshold:

    "at conf = T, IoU >= 0.5, box-level miss rate = FN / N_gt"

so: predict once with a very low conf, then threshold + match here.

Matching rule (documented so the number is reproducible):
  * per class, greedy over predictions by descending confidence
  * a prediction matches the highest-IoU unmatched GT of the same class if IoU >= thr
  * every unmatched prediction is a false positive; every unmatched GT is a miss

Usage:
  python eval_pr_at_conf.py --weights <best.pt> --imgsz 640 --tag baseline_640
"""
import argparse
import glob
import json
import os

import numpy as np
from ultralytics import YOLO

LABELS = r'E:\yolo\datasets\neu-det-yolo\labels\test'
IMAGES = r'E:\yolo\datasets\neu-det-yolo\images\test'
OUT = r'E:\yolo\runs\eval'
CLASSES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']


def load_gt(label_path, w, h):
    """YOLO txt -> (N,5) float array [class, x1, y1, x2, y2] in pixels."""
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


def iou_one_to_many(box, boxes):
    """IoU of a single xyxy box against (M,4) boxes -> (M,)"""
    if len(boxes) == 0:
        return np.zeros(0, dtype=np.float32)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    area_b = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None)
    union = area_a + area_b - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def evaluate(preds, gts, conf, iou_thr, only_class=None):
    """preds: list of (cls[], score[], box[][4]); gts: list of (cls[], box[][4]).

    only_class: if given, restrict both sides to that class id.
    Returns tp/fp/fn + precision/recall/miss rates.
    """
    tp = fp = fn = 0
    img_missed = 0
    n_img = 0
    for (pc, ps, pb), (gc, gb) in zip(preds, gts):
        if only_class is not None:
            pk = np.where(pc == only_class)[0]
            pc, ps, pb = pc[pk], ps[pk], pb[pk]
            gk = np.where(gc == only_class)[0]
            gc, gb = gc[gk], gb[gk]
        keep = ps >= conf
        pc, ps, pb = pc[keep], ps[keep], pb[keep]
        n_img += 1
        matched = np.zeros(len(gb), dtype=bool)
        for j in np.argsort(-ps) if len(ps) else []:
            same = np.where(gc == pc[j])[0]
            if len(same) == 0:
                fp += 1
                continue
            ious = iou_one_to_many(pb[j], gb[same])
            k = int(np.argmax(ious))
            if ious[k] >= iou_thr and not matched[same[k]]:
                matched[same[k]] = True
                tp += 1
            else:
                fp += 1
        miss = int((~matched).sum())
        fn += miss
        if miss > 0:
            img_missed += 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return {'conf': conf, 'tp': tp, 'fp': fp, 'fn': fn,
            'precision': prec, 'recall': rec,
            'box_miss_rate': 1 - rec,
            'image_miss_rate': (img_missed / n_img) if n_img else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--iou', type=float, default=0.5)
    ap.add_argument('--tag', default='baseline_640')
    ap.add_argument('--confs', default='0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6')
    ap.add_argument('--nms-iou', type=float, default=0.7, help='NMS IoU used when predicting')
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    model = YOLO(args.weights)
    files = sorted(glob.glob(os.path.join(IMAGES, '*.jpg')))
    print('=== explicit-threshold evaluation (own matcher) ===')
    print('  images=%d  iou=%.2f  nms_iou=%.2f  imgsz=%d  conf_for_predict=0.01'
          % (len(files), args.iou, args.nms_iou, args.imgsz))

    preds, gts = [], []
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        r = model.predict(f, imgsz=args.imgsz, conf=0.01, iou=args.nms_iou,
                          verbose=False, max_det=300)[0]
        h, w = r.orig_shape
        if len(r.boxes):
            preds.append((r.boxes.cls.cpu().numpy().astype(int),
                          r.boxes.conf.cpu().numpy(),
                          r.boxes.xyxy.cpu().numpy()))
        else:
            preds.append((np.zeros(0, dtype=int), np.zeros(0, dtype=np.float32),
                          np.zeros((0, 4), dtype=np.float32)))
        gt = load_gt(os.path.join(LABELS, stem + '.txt'), w, h)
        gts.append((gt[:, 0].astype(int) if len(gt) else np.zeros(0, dtype=int),
                    gt[:, 1:5] if len(gt) else np.zeros((0, 4), dtype=np.float32)))

    n_gt = int(sum(len(g[0]) for g in gts))
    print('  GT boxes on test split = %d' % n_gt)
    if n_gt == 0:
        raise SystemExit('no GT labels found - check LABELS path')

    confs = [float(c) for c in args.confs.split(',')]
    rows = []
    print('')
    print('  conf     TP    FP    FN   precision  recall   box-miss  img-miss')
    for c in confs:
        m = evaluate(preds, gts, c, args.iou)
        rows.append(m)
        print('  %.2f  %5d %5d %5d     %.4f   %.4f    %6.2f%%   %6.2f%%'
              % (c, m['tp'], m['fp'], m['fn'], m['precision'], m['recall'],
                 100 * m['box_miss_rate'], 100 * m['image_miss_rate']))

    focus = 0.25
    print('')
    print('--- per class @conf=%.2f ---' % focus)
    per_class = {}
    for ci, cname in enumerate(CLASSES):
        m = evaluate(preds, gts, focus, args.iou, only_class=ci)
        per_class[cname] = m
        print('  %-16s TP=%3d FP=%3d FN=%3d   P=%.4f R=%.4f  miss=%6.2f%%'
              % (cname, m['tp'], m['fp'], m['fn'], m['precision'], m['recall'],
                 100 * m['box_miss_rate']))

    out = {'weights': args.weights, 'imgsz': args.imgsz, 'iou': args.iou,
           'nms_iou': args.nms_iou, 'n_images': len(files), 'n_gt': n_gt,
           'rows': rows, 'per_class_at_%s' % focus: per_class}
    path = os.path.join(OUT, 'pr_at_conf_%s.json' % args.tag)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print('')
    print('  saved -> %s' % path)


if __name__ == '__main__':
    main()
