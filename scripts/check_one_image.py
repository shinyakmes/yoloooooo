# -*- coding: utf-8 -*-
"""
作业验收工具：给一张图的预测 CSV + 标准答案 XML，打印 IoU 矩阵和三格判定。

用法：
  python check_one_image.py --csv F:\\yolo\\demo\\hw\\detections.csv ^
      --xml F:\\yolo\\datasets\\NEU-DET\\ANNOTATIONS\\scratches_1.xml --conf 0.25 --iou 0.5
"""
import argparse
import csv
import os
import xml.etree.ElementTree as ET


def load_preds(path, conf):
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            c = float(r['conf'])
            if c < conf:
                continue
            rows.append({'cls': r['class'], 'conf': c,
                         'box': (float(r['x1']), float(r['y1']), float(r['x2']), float(r['y2']))})
    rows.sort(key=lambda x: -x['conf'])
    return rows


def load_gt(path):
    root = ET.parse(path).getroot()
    gts = []
    for o in root.findall('object'):
        bb = o.find('bndbox')
        gts.append({'cls': (o.findtext('name') or '').strip(),
                    'box': tuple(float(bb.findtext(k)) for k in ('xmin', 'ymin', 'xmax', 'ymax'))})
    return gts


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ab = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = aa + ab - inter
    return iw, ih, inter, union, (inter / union if union > 0 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--xml', required=True)
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--iou', type=float, default=0.5)
    args = ap.parse_args()

    preds = load_preds(args.csv, args.conf)
    gts = load_gt(args.xml)
    print('图片: %s' % os.path.basename(args.xml))
    print('阈值: conf >= %.2f, IoU >= %.2f' % (args.conf, args.iou))
    print('')
    print('--- 标准答案(GT) ---')
    for i, g in enumerate(gts):
        w = g['box'][2] - g['box'][0]
        h = g['box'][3] - g['box'][1]
        print('  GT%d  %-16s (%.0f,%.0f)-(%.0f,%.0f)  宽%.0f 高%.0f 面积%.0f'
              % (i + 1, g['cls'], g['box'][0], g['box'][1], g['box'][2], g['box'][3], w, h, w * h))
    print('')
    print('--- 模型预测（按置信度从高到低）---')
    for i, p in enumerate(preds):
        w = p['box'][2] - p['box'][0]
        h = p['box'][3] - p['box'][1]
        print('  预测%d conf=%.4f  %-16s (%.1f,%.1f)-(%.1f,%.1f)  宽%.2f 高%.2f 面积%.0f'
              % (i + 1, p['conf'], p['cls'], p['box'][0], p['box'][1], p['box'][2], p['box'][3], w, h, w * h))

    print('')
    print('--- IoU 矩阵（逐对算）---')
    header = '              ' + ''.join('  GT%-9d' % (j + 1) for j in range(len(gts)))
    print(header)
    for i, p in enumerate(preds):
        line = '  预测%d        ' % (i + 1)
        for g in gts:
            w, h, inter, union, v = iou(p['box'], g['box'])
            line += '  %8.4f ' % v
        print(line)

    print('')
    print('--- 判定（贪心：按置信度从高到低，每个 GT 只能被匹配一次）---')
    gt_taken = [False] * len(gts)
    tp = fp = 0
    for i, p in enumerate(preds):
        best_j, best_v = -1, 0.0
        for j, g in enumerate(gts):
            if g['cls'] != p['cls']:
                continue
            v = iou(p['box'], g['box'])[4]
            if v > best_v:
                best_j, best_v = j, v
        if best_j >= 0 and best_v >= args.iou and not gt_taken[best_j]:
            gt_taken[best_j] = True
            tp += 1
            print('  预测%d (conf=%.4f) -> TP 命中：匹配 GT%d，IoU=%.4f' % (i + 1, p['conf'], best_j + 1, best_v))
        elif best_j >= 0 and best_v >= args.iou and gt_taken[best_j]:
            fp += 1
            print('  预测%d (conf=%.4f) -> FP 误报：和 GT%d 重合(IoU=%.4f)但该 GT 已被更高置信度的框匹配'
                  % (i + 1, p['conf'], best_j + 1, best_v))
        elif best_j >= 0:
            fp += 1
            print('  预测%d (conf=%.4f) -> FP 误报：最接近 GT%d 但 IoU=%.4f < %.2f，不算命中'
                  % (i + 1, p['conf'], best_j + 1, best_v, args.iou))
        else:
            fp += 1
            print('  预测%d (conf=%.4f) -> FP 误报：没有同类 GT' % (i + 1, p['conf']))
    fn = 0
    for j, g in enumerate(gts):
        if not gt_taken[j]:
            fn += 1
            print('  GT%d (%s) -> FN 漏检：没有任何预测与之匹配' % (j + 1, g['cls']))
    for j, g in enumerate(gts):
        if gt_taken[j]:
            print('  GT%d (%s) -> 已命中（计入 TP）' % (j + 1, g['cls']))

    print('')
    print('--- 汇总 ---')
    print('  TP=%d  FP=%d  FN=%d   （GT 总数=%d，验算 TP+FN=%d 应等于 %d）'
          % (tp, fp, fn, len(gts), tp + fn, len(gts)))
    if tp + fp > 0:
        print('  precision(该图) = %d/%d = %.4f' % (tp, tp + fp, tp / (tp + fp)))
    if tp + fn > 0:
        print('  recall(该图)    = %d/%d = %.4f' % (tp, tp + fn, tp / (tp + fn)))


if __name__ == '__main__':
    main()
