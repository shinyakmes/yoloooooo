# -*- coding: utf-8 -*-
"""
Summarize one or more ultralytics training runs from their results.csv.

Usage:
  python summarize_run.py F:\\yolo\\runs\\baseline_640
  python summarize_run.py F:\\yolo\\runs\\baseline_640 F:\\yolo\\runs\\cmp_320

Prints per-run: best epoch (by mAP50-95), final metrics, training time, and
emits a markdown table for the README. Also writes <run>\\eval_summary.md.
"""
import os
import sys
import csv


def load(run_dir):
    path = os.path.join(run_dir, 'results.csv')
    if not os.path.exists(path):
        return None
    rows = []
    with open(path, newline='', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for r in reader:
            # ultralytics pads header names with spaces
            rows.append({(k or '').strip(): (v or '').strip() for k, v in r.items()})
    return rows


def fnum(r, key):
    try:
        return float(r.get(key, '') or 'nan')
    except ValueError:
        return float('nan')


def summarize(run_dir, rows):
    name = os.path.basename(os.path.normpath(run_dir))
    print('=' * 78)
    print('RUN: %s   epochs_logged=%d' % (name, len(rows)))
    if not rows:
        return None
    best = max(rows, key=lambda r: (fnum(r, 'metrics/mAP50-95(B)') if fnum(r, 'metrics/mAP50-95(B)') == fnum(r, 'metrics/mAP50-95(B)') else -1))
    # pick the run's own imgsz / batch from args.yaml when available
    imgsz = batch = '?'
    args_path = os.path.join(run_dir, 'args.yaml')
    if os.path.exists(args_path):
        for line in open(args_path, encoding='utf-8', errors='replace'):
            if line.startswith('imgsz:'):
                imgsz = line.split(':', 1)[1].strip()
            elif line.startswith('batch:'):
                batch = line.split(':', 1)[1].strip()

    final = rows[-1]
    # 'time' is cumulative wall-clock, so run length is its max (not the sum)
    t = max(fnum(r, 'time') for r in rows)
    out = {
        'name': name,
        'imgsz': imgsz,
        'batch': batch,
        'epochs': len(rows),
        'best_epoch': int(fnum(best, 'epoch')),
        'map50': fnum(best, 'metrics/mAP50(B)'),
        'map50_95': fnum(best, 'metrics/mAP50-95(B)'),
        'p': fnum(best, 'metrics/precision(B)'),
        'r': fnum(best, 'metrics/recall(B)'),
        'train_s': t,
    }
    print('  imgsz=%s batch=%s epochs=%d' % (out['imgsz'], out['batch'], out['epochs']))
    print('  best epoch %d : mAP50=%.4f  mAP50-95=%.4f  P=%.4f  R=%.4f'
          % (out['best_epoch'], out['map50'], out['map50_95'], out['p'], out['r']))
    print('  final epoch: mAP50=%.4f  mAP50-95=%.4f'
          % (fnum(final, 'metrics/mAP50(B)'), fnum(final, 'metrics/mAP50-95(B)')))
    print('  train+val time = %.1f min' % (t / 60.0))

    # loss trajectory, first vs last
    print('  losses (train/val box, cls, dfl): first -> last')
    for k in ('train/box_loss', 'train/cls_loss', 'train/dfl_loss',
              'val/box_loss', 'val/cls_loss', 'val/dfl_loss'):
        print('    %-16s %.4f -> %.4f' % (k, fnum(rows[0], k), fnum(final, k)))

    md = ('| %s | %s | %d | %.4f | %.4f | %.4f | %.4f | %.1f |\n'
          % (out['name'], out['imgsz'], out['best_epoch'], out['map50'], out['map50_95'],
             out['p'], out['r'], out['train_s'] / 60.0))
    with open(os.path.join(run_dir, 'eval_summary.md'), 'w', encoding='utf-8') as fh:
        fh.write('| run | imgsz | best epoch | mAP@0.5 | mAP@0.5:0.95 | precision | recall | train min |\n')
        fh.write('|---|---|---|---|---|---|---|---|\n')
        fh.write(md)
    return out


def main():
    dirs = sys.argv[1:]
    if not dirs:
        print(__doc__)
        return 1
    results = []
    for d in dirs:
        rows = load(d)
        if rows is None:
            print('SKIP %s (no results.csv)' % d)
            continue
        out = summarize(d, rows)
        if out:
            results.append(out)
    if len(results) > 1:
        print()
        print('=== MARKDOWN TABLE (paste into README) ===')
        print('| run | imgsz | best epoch | mAP@0.5 | mAP@0.5:0.95 | precision | recall | train min |')
        print('|---|---|---|---|---|---|---|---|')
        for o in results:
            print('| %s | %s | %d | %.4f | %.4f | %.4f | %.4f | %.1f |'
                  % (o['name'], o['imgsz'], o['best_epoch'], o['map50'], o['map50_95'],
                     o['p'], o['r'], o['train_s'] / 60.0))
    return 0


if __name__ == '__main__':
    sys.exit(main())
