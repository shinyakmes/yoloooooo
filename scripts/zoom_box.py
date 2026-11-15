# -*- coding: utf-8 -*-
"""把某个标注框所在区域裁剪并放大，看清它到底盖住了什么。

用法:
  python zoom_box.py --image <jpg> --box 8,184,17,196 --scale 12 --out <png>
"""
import argparse

import cv2

ap = argparse.ArgumentParser()
ap.add_argument('--image', required=True)
ap.add_argument('--box', required=True, help='x1,y1,x2,y2')
ap.add_argument('--scale', type=int, default=12)
ap.add_argument('--margin', type=int, default=6)
ap.add_argument('--out', required=True)
args = ap.parse_args()

x1, y1, x2, y2 = [float(v) for v in args.box.split(',')]
img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
h, w = img.shape

cx1 = max(0, int(x1) - args.margin)
cy1 = max(0, int(y1) - args.margin)
cx2 = min(w, int(x2) + args.margin + 1)
cy2 = min(h, int(y2) + args.margin + 1)
crop = img[cy1:cy2, cx1:cx2]

big = cv2.resize(crop, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_NEAREST)
big = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)

bx1 = int((x1 - cx1) * args.scale)
by1 = int((y1 - cy1) * args.scale)
bx2 = int((x2 - cx1) * args.scale)
by2 = int((y2 - cy1) * args.scale)
cv2.rectangle(big, (bx1, by1), (bx2, by2), (0, 255, 0), 2)

cv2.putText(big, 'GT2: %dx%d px' % (x2 - x1, y2 - y1), (bx1 + 4, max(14, by1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
cv2.putText(big, 'zoom x%d (src region %d,%d - %d,%d)' % (args.scale, cx1, cy1, cx2, cy2),
            (4, big.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
cv2.imwrite(args.out, big)
print('saved ->', args.out, big.shape)
