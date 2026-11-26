# -*- coding: utf-8 -*-
"""生成 CLAHE 增强版数据集，用于「单变量对照实验」。

设计原则（这是可比性的全部来源，别改）：
  1. **只改像素，不动划分**：图片从 images/<split> 读，写到新根的同名 split；
     **labels 原样复制**。同一个 GT、同一个测试集，所以任何指标差异都只能归因于像素处理。
  2. **脚本可重跑**：参数写进 manifest.json，任何人 clone 之后两条命令能复现同一份数据。
  3. **保留原始文件名**（只换扩展名为 .png），labels 的 stem 才能对上。

处理方式：BGR -> LAB，只对 L（亮度）通道做 CLAHE，再合回去。
  - 不动 a/b：避免把钢材的颜色噪声也放大；
  - 输出 .png 无损：避免二次 JPEG 压缩在低对比纹理上引入块效应（那会污染结论）。

用法：
    python make_clahe_dataset.py                 # 默认 clip=2.0 tile=8
    python make_clahe_dataset.py --clip 3.0      # 想试更激进的对比度
"""
import argparse
import glob
import hashlib
import json
import os
import shutil
import sys

import cv2
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE = os.environ.get('YOLO_ROOT', r'E:\yolo')
SRC = os.path.join(BASE, 'datasets', 'neu-det-yolo')
SPLITS = ['train', 'val', 'test']
NAMES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 16), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=SRC)
    ap.add_argument('--dst', default=os.path.join(BASE, 'datasets', 'neu-det-yolo-clahe'))
    ap.add_argument('--clip', type=float, default=2.0, help='CLAHE clipLimit')
    ap.add_argument('--tile', type=int, default=8, help='CLAHE tileGridSize（正方形）')
    args = ap.parse_args()

    if os.path.abspath(args.src) == os.path.abspath(args.dst):
        raise SystemExit('dst 不能等于 src')

    clahe = cv2.createCLAHE(clipLimit=args.clip, tileGridSize=(args.tile, args.tile))
    stats = {}
    total_in, total_out = [], []

    for split in SPLITS:
        src_img = os.path.join(args.src, 'images', split)
        src_lbl = os.path.join(args.src, 'labels', split)
        dst_img = os.path.join(args.dst, 'images', split)
        dst_lbl = os.path.join(args.dst, 'labels', split)
        os.makedirs(dst_img, exist_ok=True)
        os.makedirs(dst_lbl, exist_ok=True)

        files = sorted(glob.glob(os.path.join(src_img, '*.jpg')))
        n_done = 0
        for f in files:
            stem = os.path.splitext(os.path.basename(f))[0]
            img = cv2.imread(f, cv2.IMREAD_COLOR)
            if img is None:
                raise SystemExit('读不到: %s' % f)
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            assert out.shape == img.shape, (out.shape, img.shape)
            cv2.imwrite(os.path.join(dst_img, stem + '.png'), out,
                        [cv2.IMWRITE_PNG_COMPRESSION, 3])
            total_in.append(img.mean())
            total_out.append(out.mean())
            n_done += 1

        # labels 原样复制（逐字节相同，后面会校验 md5）
        lbl_files = sorted(glob.glob(os.path.join(src_lbl, '*.txt')))
        for f in lbl_files:
            shutil.copy2(f, os.path.join(dst_lbl, os.path.basename(f)))

        stats[split] = {'images_in': len(files), 'images_out': n_done,
                        'labels_in': len(lbl_files),
                        'labels_out': len(glob.glob(os.path.join(dst_lbl, '*.txt')))}
        print('  %-6s images %d -> %d | labels %d -> %d'
              % (split, stats[split]['images_in'], n_done,
                 stats[split]['labels_in'], stats[split]['labels_out']))

    # ---------- 校验：labels 一个字节都没变 ----------
    bad = 0
    for split in SPLITS:
        for f in glob.glob(os.path.join(args.src, 'labels', split, '*.txt')):
            dstf = os.path.join(args.dst, 'labels', split, os.path.basename(f))
            if not os.path.exists(dstf) or md5(f) != md5(dstf):
                bad += 1
    print('\n  labels md5 不一致的文件数: %d %s' % (bad, '(应为 0)' if bad == 0 else '⚠️'))

    # ---------- 写 data.yaml ----------
    yml = ['# NEU-DET + CLAHE (clipLimit=%s, tile=%dx%d)，由 make_clahe_dataset.py 生成'
           % (args.clip, args.tile, args.tile),
           '# 划分与 labels 与原始 neu-det-yolo 完全一致，只改像素 -> 可做单变量对照',
           'train: images/train', 'val: images/val', 'test: images/test', '',
           'names:']
    for i, n in enumerate(NAMES):
        yml.append('  %d: %s' % (i, n))
    with open(os.path.join(args.dst, 'data.yaml'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(yml) + '\n')

    manifest = {
        'src': args.src, 'dst': args.dst, 'color_space': 'BGR->LAB, CLAHE on L only',
        'clip_limit': args.clip, 'tile_grid': [args.tile, args.tile],
        'output_format': 'png (lossless)', 'splits': stats,
        'mean_brightness_in': float(np.mean(total_in)),
        'mean_brightness_out': float(np.mean(total_out)),
        'labels_md5_mismatch': bad,
    }
    with open(os.path.join(args.dst, 'manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print('\n  平均亮度: 前 %.2f -> 后 %.2f  (CLAHE 是直方图均衡，均值应基本不动、对比度上升)'
          % (np.mean(total_in), np.mean(total_out)))
    print('  data.yaml -> %s' % os.path.join(args.dst, 'data.yaml'))
    print('  manifest  -> %s' % os.path.join(args.dst, 'manifest.json'))


if __name__ == '__main__':
    main()
