# -*- coding: utf-8 -*-
"""
D2: convert NEU-DET (VOC XML) -> YOLO format with a stratified, seeded split.

Why our own converter instead of a ready-made YOLO copy of this dataset:
  - the original annotation format is what we can explain and defend;
  - the split is explicit, reproducible (seed) and stratified by class,
    which is what the README/report needs to state.

Layout produced:
  F:\\yolo\\datasets\\neu-det-yolo\\
    data.yaml
    images\\{train,val,test}\\*.jpg
    labels\\{train,val,test}\\*.txt
    convert_stats.txt

label line format: <class_id> <cx> <cy> <w> <h>   (all normalized to 0..1)
"""
import os
import random
import shutil
import time
import collections
import xml.etree.ElementTree as ET

SRC = r'E:\yolo\datasets\NEU-DET'
IMG_DIR = os.path.join(SRC, 'IMAGES')
ANN_DIR = os.path.join(SRC, 'ANNOTATIONS')
DST = r'E:\yolo\datasets\neu-det-yolo'

CLASSES = ['crazing', 'inclusion', 'patches',
           'pitted_surface', 'rolled-in_scale', 'scratches']
SPLIT = (0.8, 0.1, 0.1)      # train / val / test
SEED = 20261021

LOG = []


def out(msg):
    LOG.append(msg)
    print(msg, flush=True)


def parse(xml_path):
    """-> (W, H, [(cls_id, cx, cy, bw, bh), ...], notes)"""
    root = ET.parse(xml_path).getroot()
    size = root.find('size')
    if size is None:
        raise ValueError('no <size>')
    W = int(float(size.findtext('width', '0')))
    H = int(float(size.findtext('height', '0')))
    if W <= 0 or H <= 0:
        raise ValueError('bad size %dx%d' % (W, H))
    boxes = []
    notes = []
    for o in root.findall('object'):
        name = (o.findtext('name') or '').strip()
        if name not in CLASSES:
            notes.append('unknown class %r' % name)
            continue
        bb = o.find('bndbox')
        if bb is None:
            notes.append('no bndbox')
            continue
        x1 = float(bb.findtext('xmin', '0'))
        y1 = float(bb.findtext('ymin', '0'))
        x2 = float(bb.findtext('xmax', '0'))
        y2 = float(bb.findtext('ymax', '0'))
        # clamp into image (a few boxes touch the border by 1 px)
        x1 = max(0.0, min(x1, W))
        x2 = max(0.0, min(x2, W))
        y1 = max(0.0, min(y1, H))
        y2 = max(0.0, min(y2, H))
        if x2 - x1 < 1 or y2 - y1 < 1:
            notes.append('degenerate box dropped')
            continue
        cx = (x1 + x2) / 2.0 / W
        cy = (y1 + y2) / 2.0 / H
        bw = (x2 - x1) / W
        bh = (y2 - y1) / H
        boxes.append((CLASSES.index(name), cx, cy, bw, bh))
    return W, H, boxes, notes


def main():
    t0 = time.time()
    out('=== VOC -> YOLO conversion ===')
    out('src = %s' % SRC)
    out('dst = %s' % DST)

    # 1) group image stems by class, from the ANNOTATIONS folder
    by_class = collections.defaultdict(list)
    for a in sorted(os.listdir(ANN_DIR)):
        if not a.lower().endswith('.xml'):
            continue
        stem = os.path.splitext(a)[0]
        cls = stem.rsplit('_', 1)[0]
        if cls not in CLASSES:
            out('  SKIP unknown prefix: %s' % a)
            continue
        by_class[cls].append(stem)

    for c in CLASSES:
        out('  class %-16s %d annotations' % (c, len(by_class[c])))

    # 2) stratified split per class
    rng = random.Random(SEED)
    assign = {}
    for c in CLASSES:
        stems = sorted(by_class[c])
        rng.shuffle(stems)
        n = len(stems)
        n_train = int(round(n * SPLIT[0]))
        n_val = int(round(n * SPLIT[1]))
        for i, s in enumerate(stems):
            if i < n_train:
                split = 'train'
            elif i < n_train + n_val:
                split = 'val'
            else:
                split = 'test'
            assign[s] = split
    out('  seed = %d  split(train/val/test) = %.1f/%.1f/%.1f' % (SEED, SPLIT[0], SPLIT[1], SPLIT[2]))

    # 3) build dirs
    for split in ('train', 'val', 'test'):
        for sub in ('images', 'labels'):
            os.makedirs(os.path.join(DST, sub, split), exist_ok=True)

    # 4) convert + copy
    stats = collections.Counter()
    per_split_class = collections.Counter()
    failures = []
    n_box = 0
    for stem in sorted(assign):
        img_src = None
        for ext in ('.jpg', '.jpeg', '.png'):
            p = os.path.join(IMG_DIR, stem + ext)
            if os.path.exists(p):
                img_src = p
                break
        if img_src is None:
            failures.append('%s: image missing' % stem)
            continue
        try:
            W, H, boxes, notes = parse(os.path.join(ANN_DIR, stem + '.xml'))
        except Exception as e:
            failures.append('%s: %s' % (stem, e))
            continue
        if not boxes:
            failures.append('%s: no usable box (%s)' % (stem, '; '.join(notes)))
            continue
        split = assign[stem]
        shutil.copy2(img_src, os.path.join(DST, 'images', split, stem + '.jpg'))
        with open(os.path.join(DST, 'labels', split, stem + '.txt'), 'w', encoding='utf-8') as f:
            for cid, cx, cy, bw, bh in boxes:
                f.write('%d %.6f %.6f %.6f %.6f\n' % (cid, cx, cy, bw, bh))
        n_box += len(boxes)
        stats[split] += 1
        cls = stem.rsplit('_', 1)[0]
        per_split_class[(split, cls)] += 1

    # 5) data.yaml (forward slashes are safer for ultralytics)
    yaml_path = os.path.join(DST, 'data.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write('# NEU-DET in YOLO format, generated by convert_voc_to_yolo.py\n')
        f.write('path: %s\n' % DST.replace('\\', '/'))
        f.write('train: images/train\n')
        f.write('val: images/val\n')
        f.write('test: images/test\n')
        f.write('\nnames:\n')
        for i, c in enumerate(CLASSES):
            f.write('  %d: %s\n' % (i, c))

    out('')
    out('--- result ---')
    for split in ('train', 'val', 'test'):
        out('  %-5s images = %d' % (split, stats[split]))
    out('  total images converted = %d   boxes = %d' % (sum(stats.values()), n_box))
    out('')
    out('--- per class per split ---')
    for c in CLASSES:
        out('  %-16s train=%3d val=%3d test=%3d' % (
            c, per_split_class[('train', c)], per_split_class[('val', c)], per_split_class[('test', c)]))
    if failures:
        out('')
        out('--- skipped (%d) ---' % len(failures))
        for msg in failures[:20]:
            out('  ' + msg)
    out('')
    out('data.yaml -> %s' % yaml_path)
    out('elapsed = %.1fs' % (time.time() - t0))
    out('=== CONVERT DONE ===')

    with open(os.path.join(DST, 'convert_stats.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(LOG))


if __name__ == '__main__':
    main()
