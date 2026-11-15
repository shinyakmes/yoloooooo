# -*- coding: utf-8 -*-
"""
D8: inference program for the NEU-DET steel defect model.

Supports image / folder / video / webcam, saves annotated output and a CSV of
detections, and reports end-to-end per-frame latency (the number we quote).

Examples
--------
  # single image
  python detect.py --weights F:\\yolo\\runs\\baseline_640\\weights\\best.pt ^
                   --source F:\\yolo\\datasets\\NEU-DET\\IMAGES\\crazing_1.jpg

  # whole test split, save csv + images
  python detect.py --weights ... --source F:\\yolo\\datasets\\neu-det-yolo\\images\\test ^
                   --save-csv --out F:\\yolo\\demo\\test_pred

  # video
  python detect.py --weights ... --source demo.mp4 --out F:\\yolo\\demo\\video
"""
import argparse
import csv
import os
import time

import cv2
import numpy as np
from ultralytics import YOLO

IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
VID_EXT = ('.mp4', '.avi', '.mov', '.mkv')


def iter_sources(source):
    """Yield (kind, path_or_index) items."""
    if source.isdigit():
        yield 'webcam', int(source)
        return
    if os.path.isdir(source):
        for name in sorted(os.listdir(source)):
            if name.lower().endswith(IMG_EXT):
                yield 'image', os.path.join(source, name)
        return
    if not os.path.exists(source):
        raise SystemExit('source not found: %s' % source)
    if source.lower().endswith(VID_EXT):
        yield 'video', source
    else:
        yield 'image', source


def run(args):
    model = YOLO(args.weights)
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'detections.csv') if args.save_csv else None
    csv_f = open(csv_path, 'w', newline='', encoding='utf-8') if csv_path else None
    writer = csv.writer(csv_f) if csv_f else None
    if writer:
        writer.writerow(['source', 'frame_or_file', 'class', 'conf', 'x1', 'y1', 'x2', 'y2'])

    # warm up: the first inference pays CUDA/cuDNN autotune and model build cost
    # (measured ~3 s), which would otherwise dominate the latency statistics.
    model.predict(np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8),
                  imgsz=args.imgsz, conf=args.conf, verbose=False)

    n_img = 0
    n_obj = 0
    lat = []

    for kind, src in iter_sources(args.source):
        if kind == 'video':
            cap = cv2.VideoCapture(src)
            fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            vw = cv2.VideoWriter(os.path.join(out_dir, 'annotated.mp4'),
                                 cv2.VideoWriter_fourcc(*'mp4v'), fps_in, (w, h))
            idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                t0 = time.perf_counter()
                res = model.predict(frame, imgsz=args.imgsz, conf=args.conf, iou=args.nms_iou, verbose=False)[0]
                lat.append((time.perf_counter() - t0) * 1000)
                vw.write(res.plot())
                n_obj += len(res.boxes)
                if writer:
                    for b in res.boxes:
                        x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
                        writer.writerow([src, idx, res.names[int(b.cls)], '%.4f' % float(b.conf), x1, y1, x2, y2])
                idx += 1
            cap.release()
            vw.release()
            n_img += idx
            print('video %s: %d frames -> annotated.mp4' % (os.path.basename(src), idx))
            continue

        img = cv2.imread(src)
        if img is None:
            print('  cannot read %s' % src)
            continue
        t0 = time.perf_counter()
        res = model.predict(img, imgsz=args.imgsz, conf=args.conf, iou=args.nms_iou,
                            verbose=False)[0]
        dt = (time.perf_counter() - t0) * 1000
        lat.append(dt)
        n_img += 1
        n_obj += len(res.boxes)
        base = os.path.splitext(os.path.basename(src))[0]
        if args.save_images:
            cv2.imwrite(os.path.join(out_dir, base + '_pred.jpg'), res.plot())
        if writer:
            for b in res.boxes:
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
                writer.writerow([src, base, res.names[int(b.cls)], '%.4f' % float(b.conf), x1, y1, x2, y2])
        if args.verbose:
            names = [res.names[int(b.cls)] for b in res.boxes]
            print('  %-22s %-38s %.1f ms' % (base, ','.join(names) or '(none)', dt))

    if csv_f:
        csv_f.close()

    if lat:
        lat_sorted = sorted(lat)
        print('')
        print('--- summary ---')
        print('  inputs        = %d' % n_img)
        print('  detections    = %d' % n_obj)
        print('  latency mean  = %.1f ms/frame' % (sum(lat) / len(lat)))
        print('  latency p50   = %.1f ms' % lat_sorted[len(lat_sorted) // 2])
        print('  latency p95   = %.1f ms' % lat_sorted[min(len(lat_sorted) - 1, int(0.95 * len(lat_sorted)))])
        print('  throughput    = %.1f FPS' % (1000.0 / (sum(lat) / len(lat))))
    print('  output dir    = %s' % out_dir)
    if csv_path:
        print('  csv           = %s' % csv_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--source', required=True, help='image | folder | video | webcam index')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--nms-iou', type=float, default=0.7, help='NMS IoU threshold: lower = more aggressive de-duplication')
    ap.add_argument('--out', default=r'E:\yolo\demo\out')
    ap.add_argument('--save-images', action='store_true')
    ap.add_argument('--save-csv', action='store_true')
    ap.add_argument('--verbose', action='store_true')
    run(ap.parse_args())


if __name__ == '__main__':
    main()

