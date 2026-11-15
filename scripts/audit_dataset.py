# -*- coding: utf-8 -*-
"""
D2: NEU-DET dataset audit.

Reads the raw VOC-style dataset (IMAGES/*.jpg + ANNOTATIONS/*.xml) and prints
statistics that we will need for (a) the README, (b) choosing imgsz, and
(c) spotting problems before spending GPU time.

Input : F:\\yolo\\datasets\\NEU-DET
Output: console + F:\\yolo\\datasets\\audit_report.txt
"""
import os
import time
import collections
import xml.etree.ElementTree as ET

ROOT = r'E:\yolo\datasets\NEU-DET'
IMG_DIR = os.path.join(ROOT, 'IMAGES')
ANN_DIR = os.path.join(ROOT, 'ANNOTATIONS')
REPORT = r'E:\yolo\datasets\audit_report.txt'

LINES = []


def out(msg):
    LINES.append(msg)
    print(msg, flush=True)


def main():
    t0 = time.time()
    out('=== NEU-DET AUDIT ===')
    out('root = %s' % ROOT)

    imgs = sorted(f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    anns = sorted(f for f in os.listdir(ANN_DIR) if f.lower().endswith('.xml'))
    out('images = %d   annotations = %d' % (len(imgs), len(anns)))

    cls_counts = collections.Counter()
    obj_per_img = collections.Counter()
    sizes = collections.Counter()
    box_areas = []
    tiny = 0            # boxes < 0.5% of image area
    big = 0             # boxes > 80% of image area
    out_of_bounds = 0
    zero_area = 0
    per_img_boxes = []

    for a in anns:
        try:
            root = ET.parse(os.path.join(ANN_DIR, a)).getroot()
        except Exception as e:
            out('  PARSE FAIL %s: %s' % (a, e))
            continue
        size = root.find('size')
        if size is None:
            continue
        W = int(float(size.findtext('width', '0')))
        H = int(float(size.findtext('height', '0')))
        sizes['%dx%d' % (W, H)] += 1
        objs = root.findall('object')
        obj_per_img[len(objs)] += 1
        per_img_boxes.append(len(objs))
        for o in objs:
            name = (o.findtext('name') or '?').strip()
            cls_counts[name] += 1
            bb = o.find('bndbox')
            if bb is None:
                continue
            x1 = float(bb.findtext('xmin', '0'))
            y1 = float(bb.findtext('ymin', '0'))
            x2 = float(bb.findtext('xmax', '0'))
            y2 = float(bb.findtext('ymax', '0'))
            w = x2 - x1
            h = y2 - y1
            if w <= 0 or h <= 0:
                zero_area += 1
                continue
            if x1 < 0 or y1 < 0 or x2 > W + 0.5 or y2 > H + 0.5:
                out_of_bounds += 1
            frac = (w * h) / float(W * H) if W and H else 0
            box_areas.append(frac)
            if frac < 0.005:
                tiny += 1
            if frac > 0.80:
                big += 1

    out('')
    out('--- image sizes ---')
    for k, v in sizes.most_common(6):
        out('  %s : %d' % (k, v))

    out('')
    out('--- class distribution (bbox instances) ---')
    for k, v in sorted(cls_counts.items()):
        out('  %-16s %4d' % (k, v))

    out('')
    out('--- objects per image ---')
    for k in sorted(obj_per_img):
        out('  %d obj : %d images' % (k, obj_per_img[k]))

    box_areas.sort()
    n = len(box_areas)

    def pct(p):
        return box_areas[min(n - 1, int(p * n))] if n else 0

    out('')
    out('--- bbox area as fraction of image (n=%d) ---' % n)
    out('  min=%.4f  p10=%.4f  p50=%.4f  p90=%.4f  max=%.4f' %
        (box_areas[0] if n else 0, pct(0.10), pct(0.50), pct(0.90), box_areas[-1] if n else 0))
    out('  boxes < 0.5%% of image : %d (%.1f%%)' % (tiny, 100.0 * tiny / max(1, n)))
    out('  boxes > 80%% of image  : %d (%.1f%%)' % (big, 100.0 * big / max(1, n)))
    out('  degenerate (w/h<=0)   : %d' % zero_area)
    out('  out of bounds         : %d' % out_of_bounds)

    out('')
    out('--- keys for decision making ---')
    out('  median objects/image = %.2f' % (sum(per_img_boxes) / max(1, len(per_img_boxes))))
    out('  images at 200x200    -> imgsz=640 upscales 3.2x; imgsz=320 upscales 1.6x')
    out('  elapsed = %.1fs' % (time.time() - t0))
    out('=== AUDIT DONE ===')

    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(LINES))


if __name__ == '__main__':
    main()
