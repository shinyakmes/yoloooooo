# -*- coding: utf-8 -*-
"""
Latency benchmark that separates compute from per-call overhead.

A naive `for _ in range(100): net(x)` loop mostly measures Python/CUDA launch
overhead on a small model, which hides the real imgsz difference (320 vs 640
came out almost identical in the first pass). This script reports:

  * single-stream latency (batch=1) measured with CUDA events
  * saturated throughput (batch=16/32) in images/s  <- the honest "speed" number
  * both for imgsz 320 / 416 / 640 so the scaling is visible

Usage:
  python bench_latency.py --weights E:\yolo\runs\baseline_640\weights\best.pt --imgsz 640,320,416
"""
import argparse
import json
import os
import time

import torch

from ultralytics import YOLO

OUT = r'E:\yolo\runs\eval'


def bench(net, imgsz, batch, iters=50, warmup=10):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.zeros(batch, 3, imgsz, imgsz, device=dev)
    with torch.no_grad():
        for _ in range(warmup):
            net(x)
        if dev == 'cuda':
            torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True) if dev == 'cuda' else None
        end = torch.cuda.Event(enable_timing=True) if dev == 'cuda' else None
        if start:
            start.record()
        for _ in range(iters):
            net(x)
        if end:
            end.record()
            torch.cuda.synchronize()
            ms = start.elapsed_time(end) / iters
        else:
            t0 = time.perf_counter()
            for _ in range(iters):
                net(x)
            ms = (time.perf_counter() - t0) / iters * 1000
    return ms, (batch * 1000.0 / ms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--imgsz', default='640,416,320')
    ap.add_argument('--tag', default='latency')
    args = ap.parse_args()
    sizes = [int(s) for s in args.imgsz.split(',')]

    os.makedirs(OUT, exist_ok=True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    model = YOLO(args.weights)
    net = model.model.eval()
    if torch.cuda.is_available():
        net = net.cuda()

    print('=== latency benchmark ===')
    print('  gpu: %s' % (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'))
    print('  model: %s' % args.weights)
    results = []
    for imgsz in sizes:
        for batch in (1, 16):
            ms, ips = bench(net, imgsz, batch)
            results.append({'imgsz': imgsz, 'batch': batch, 'ms_per_batch': round(ms, 2),
                            'ms_per_image': round(ms / batch, 2), 'img_per_s': round(ips, 1)})
            print('  imgsz=%-4d batch=%-3d : %7.2f ms/batch = %6.2f ms/img  | %6.1f img/s'
                  % (imgsz, batch, ms, ms / batch, ips))

    path = os.path.join(OUT, 'bench_%s.json' % args.tag)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'weights': args.weights, 'gpu': torch.cuda.get_device_name(0), 'rows': results}, f, indent=2)
    print('  saved -> %s' % path)


if __name__ == '__main__':
    main()
