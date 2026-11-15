# -*- coding: utf-8 -*-
"""
D5: evaluate a trained model on the held-out TEST split and produce the three
numbers the resume needs, measured the same way every time:

  1. mAP@0.5 (+ mAP@0.5:0.95, per-class AP)
  2. single-image inference latency (ms) -- pure model forward, GPU, warmup + N runs
  3. test-set miss rate -- reported two ways:
       box-level  : 1 - recall  (a GT box with no matching prediction counts as missed)
       image-level: share of test images where NO GT box got matched

Usage:
  python eval_test.py --weights F:\\yolo\\runs\\baseline_640\\weights\\best.pt --imgsz 640
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from ultralytics import YOLO

DATA = r'E:\yolo\datasets\neu-det-yolo\data.yaml'
OUT_DIR = r'E:\yolo\runs\eval'
CLASSES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']


def measure_latency(model, imgsz, runs=100, warmup=20):
    """Pure forward latency on the GPU with a dummy batch of 1."""
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = model.model.to(dev).eval()
    x = torch.zeros(1, 3, imgsz, imgsz, device=dev)
    with torch.no_grad():
        for _ in range(warmup):
            net(x)
        if dev == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(runs):
            net(x)
        if dev == 'cuda':
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / runs * 1000.0
    return dt


def match_stats(model, imgsz, conf, iou):
    """Box-level and image-level miss stats over the test split."""
    res = model.val(data=DATA, split='test', imgsz=imgsz, conf=conf, iou=iou,
                    plots=False, verbose=False)
    box = res.box
    total_gt = int(box.nt_per_class.sum()) if hasattr(box, 'nt_per_class') else None
    return res, total_gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--iou', type=float, default=0.5)
    ap.add_argument('--tag', default='')
    ap.add_argument('--runs', type=int, default=100)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    tag = args.tag or ('%s_%d' % (os.path.basename(os.path.dirname(os.path.dirname(args.weights))), args.imgsz))
    out = {'weights': args.weights, 'imgsz': args.imgsz, 'conf': args.conf, 'iou': args.iou}

    print('=== EVAL %s ===' % tag, flush=True)
    model = YOLO(args.weights)

    # --- 1) metrics on test split -------------------------------------------
    m = model.val(data=DATA, split='test', imgsz=args.imgsz, conf=0.001, iou=0.6,
                  plots=True, verbose=False)
    out['map50'] = float(m.box.map50)
    out['map50_95'] = float(m.box.map)
    out['precision'] = float(m.box.mp)
    out['recall'] = float(m.box.mr)
    out['per_class_ap50'] = [None if v != v else float(v) for v in m.box.ap50]
    out['per_class_names'] = list(m.names.values()) if isinstance(m.names, dict) else list(m.names)
    print('  mAP@0.5      = %.4f' % out['map50'])
    print('  mAP@0.5:0.95 = %.4f' % out['map50_95'])
    print('  precision    = %.4f   recall = %.4f' % (out['precision'], out['recall']))
    for name, ap in zip(out['per_class_names'], out['per_class_ap50']):
        print('    %-16s AP@0.5 = %s' % (name, ('%.4f' % ap) if ap is not None else 'n/a'))

    # --- 2) latency ---------------------------------------------------------
    # NOTE: miss rate is deliberately NOT computed here. ultralytics' box.mp/mr
    # is the value at the max-F1 point of the PR curve, NOT at the conf you pass,
    # so it cannot answer "what is the miss rate at conf=T". Use
    # scripts/eval_pr_at_conf.py for that (own matcher, explicit threshold).
    print('  (miss rate at a chosen conf: see scripts/eval_pr_at_conf.py)')

    # --- 3) inference latency ----------------------------------------------
    lat = measure_latency(model, args.imgsz, runs=args.runs)
    out['latency_ms_forward'] = lat
    print('  forward latency = %.2f ms/image (@%d, %d-run mean, batch=1)' % (lat, args.imgsz, args.runs))

    # end-to-end incl. preprocess/postprocess, for comparison
    t0 = time.perf_counter()
    _ = model.predict(source=r'E:\yolo\datasets\NEU-DET\IMAGES', imgsz=args.imgsz,
                      conf=args.conf, verbose=False, stream=False)
    e2e = (time.perf_counter() - t0)
    out['e2e_seconds_all_test_imgs'] = e2e
    print('  end-to-end total for the source folder = %.1f s' % e2e)

    out['torch'] = torch.__version__
    out['gpu'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
    path = os.path.join(OUT_DIR, 'eval_%s.json' % tag)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print('  saved -> %s' % path)
    print('=== EVAL DONE ===')


if __name__ == '__main__':
    main()
