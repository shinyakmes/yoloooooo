# -*- coding: utf-8 -*-
"""
Build a demo video from TEST images only (never train/val), synthesising a
"line stream" for the 30-second demo. Documented in the README so nobody
mistakes it for real footage.

Usage:
  python make_demo_video.py [--fps 6] [--side 500] [--out F:\\yolo\\demo\\line_sim.mp4]
"""
import argparse
import os


# 数据根目录：可用环境变量 YOLO_ROOT 覆盖
BASE = os.environ.get('YOLO_ROOT', r'E:\yolo')

import cv2

SRC = os.path.join(BASE, 'datasets', 'neu-det-yolo', 'images', 'test')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=SRC)
    ap.add_argument('--out', default=os.path.join(BASE, 'demo', 'line_sim.mp4'))
    ap.add_argument('--fps', type=float, default=6.0)
    ap.add_argument('--side', type=int, default=500)
    args = ap.parse_args()

    names = sorted(n for n in os.listdir(args.src) if n.lower().endswith(('.jpg', '.png')))
    if not names:
        raise SystemExit('no images in %s' % args.src)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    writer = None
    n = 0
    for name in names:
        img = cv2.imread(os.path.join(args.src, name))
        if img is None:
            continue
        frame = cv2.resize(img, (args.side, args.side), interpolation=cv2.INTER_CUBIC)
        if writer is None:
            writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*'mp4v'),
                                     args.fps, (args.side, args.side))
        # a label bar so the viewer knows which frame is which
        cv2.rectangle(frame, (0, 0), (args.side, 22), (0, 0, 0), -1)
        cv2.putText(frame, 'TEST %03d/%03d  %s' % (n + 1, len(names), name),
                    (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(frame)
        n += 1
    if writer:
        writer.release()
    print('frames = %d  fps = %.1f  duration = %.1f s' % (n, args.fps, n / args.fps))
    print('saved -> %s' % args.out)


if __name__ == '__main__':
    main()
