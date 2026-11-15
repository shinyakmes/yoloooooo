# -*- coding: utf-8 -*-
"""把一张图的 GT 框和预测框画在一起，用于教学对照。

用法:
  python draw_boxes_figure.py --image <jpg> --csv <detections.csv> --xml <annotation.xml> --out <png>
"""
import argparse
import csv
import xml.etree.ElementTree as ET

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True)
    ap.add_argument('--csv', required=True)
    ap.add_argument('--xml', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)

    gts = []
    root = ET.parse(args.xml).getroot()
    for o in root.findall('object'):
        bb = o.find('bndbox')
        gts.append(tuple(float(bb.findtext(k)) for k in ('xmin', 'ymin', 'xmax', 'ymax')))

    preds = []
    with open(args.csv, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            preds.append((float(r['x1']), float(r['y1']), float(r['x2']), float(r['y2']), float(r['conf'])))

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(img, cmap='gray', vmin=0, vmax=255)

    for i, g in enumerate(gts):
        ax.add_patch(Rectangle((g[0], g[1]), g[2] - g[0], g[3] - g[1],
                               fill=False, edgecolor='lime', lw=3,
                               label='GT%d 人工标注 (%.0f,%.0f)-(%.0f,%.0f)  %.0f×%.0f'
                                     % (i + 1, g[0], g[1], g[2], g[3], g[2] - g[0], g[3] - g[1])))
        ax.text(g[0], g[1] - 3, 'GT%d' % (i + 1), color='lime', fontsize=11, weight='bold')

    colors = ['cyan', 'orange', 'magenta', 'yellow']
    for i, p in enumerate(preds):
        c = colors[i % len(colors)]
        ax.add_patch(Rectangle((p[0], p[1]), p[2] - p[0], p[3] - p[1],
                               fill=False, edgecolor=c, lw=2, ls='--',
                               label='预测%d conf=%.3f  %.1f×%.1f'
                                     % (i + 1, p[4], p[2] - p[0], p[3] - p[1])))
        ax.text(p[2] + 2, p[1] + 8, 'P%d' % (i + 1), color=c, fontsize=11, weight='bold')

    ax.set_title('绿实线=人工标注(GT)   虚线=模型预测\n注意左下角那个 GT2 只有 9×12 像素，没有预测碰到它 → 漏检',
                 fontsize=10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.06), fontsize=8, ncol=1)
    ax.set_xlabel('x 像素', fontsize=9)
    ax.set_ylabel('y 像素', fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out, dpi=110, bbox_inches='tight')
    print('saved ->', args.out)
    print('GT 框数=%d  预测框数=%d' % (len(gts), len(preds)))


if __name__ == '__main__':
    main()
