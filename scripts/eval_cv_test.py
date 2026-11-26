# -*- coding: utf-8 -*-
"""交叉验证协议 v2 · 评测与汇总：**每折模型都在同一个测试集上评一次**

为什么要这样做（这是协议 v2 的核心）：

  旧做法（summarize_kfold.py）只读每折自己的 val 指标。但 5 折的 val 集合是**不同的**，
  所以那 5 个数不可配对 —— 只能看"波动有多大"，没法说"改动是否有效"。

  新做法：5 折模型全部在**同一个固定的 175 张测试集**上评。于是得到 5 个可配对的值：
    - 均值 ± 标准差   → 这就是"噪声底"，即"同样配置下指标自然会晃多少"
    - 2 × 标准差      → 判断"某改动是否有效"的参考线（差异小于它，就不要声称有效）
    - 因为测试集固定，两个配方（比如 baseline vs CLAHE）之间可以做**配对比较**
      （同一折、同一批图，逐折做差，再用符号检验/配对 t 检验）

用法：
    python eval_cv_test.py --prefix cv
    python eval_cv_test.py --prefix cv --verify        # 只做折划分的完整性校验
"""
import argparse
import csv
import json
import os
import statistics as st
import sys

import numpy as np
from ultralytics import YOLO

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE = os.environ.get('YOLO_ROOT', r'E:\yolo')
DATA = os.path.join(BASE, 'datasets', 'neu-det-yolo', 'data.yaml')
KFOLD = os.path.join(BASE, 'datasets', 'kfold')
OUT = os.path.join(BASE, 'runs', 'eval')
CLASSES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']


def verify_folds(k=5):
    """校验：各折 val 集互不重叠、并集 = train+val 全集、test 未混入。"""
    print('=== 折划分完整性校验 ===')
    seen = {}
    dup = 0
    per_fold = {}
    for f in range(k):
        vd = os.path.join(KFOLD, 'f%d' % f, 'images', 'val')
        tr = os.path.join(KFOLD, 'f%d' % f, 'images', 'train')
        val = {os.path.splitext(x)[0] for x in os.listdir(vd)} if os.path.isdir(vd) else set()
        trn = {os.path.splitext(x)[0] for x in os.listdir(tr)} if os.path.isdir(tr) else set()
        per_fold[f] = (len(trn), len(val))
        overlap = val & trn
        if overlap:
            print('  ⚠️ f%d: train 与 val 有 %d 张重叠！' % (f, len(overlap)))
        for s in val:
            if s in seen:
                dup += 1
            seen[s] = f
        print('  f%d: train=%4d val=%4d  (train∩val=%d)' % (f, len(trn), len(val), len(overlap)))

    src = set()
    for split in ('train', 'val'):
        d = os.path.join(BASE, 'datasets', 'neu-det-yolo', 'images', split)
        src |= {os.path.splitext(x)[0] for x in os.listdir(d)}
    test = {os.path.splitext(x)[0]
            for x in os.listdir(os.path.join(BASE, 'datasets', 'neu-det-yolo', 'images', 'test'))}

    seen_set = set(seen)
    print('')
    print('  train+val 全集      = %d' % len(src))
    print('  各折 val 并集       = %d' % len(seen_set))
    print('  被验证超过一次的图  = %d %s' % (dup, '（应为 0）' if dup == 0 else '⚠️'))
    print('  缺漏（全集 - 并集） = %d %s'
          % (len(src - seen_set), '（应为 0）' if not (src - seen_set) else '⚠️'))
    print('  test 混入折里的图   = %d %s'
          % (len(test & seen_set), '（应为 0）' if not (test & seen_set) else '⚠️'))
    ok = (dup == 0) and (not (src - seen_set)) and (not (test & seen_set))
    print('  结论：%s' % ('✅ 划分干净，可以当协议用' if ok else '❌ 划分有问题，先修 make_kfold.py'))
    return ok


def best_val_mAP(run_dir):
    """从 results.csv 读该折自身 val 上的 best mAP@0.5（仅供参考，跨折不可配对）。

    注意：**必须按列名取值，不能按列号**。results.csv 的列序是
      epoch, time, train/box_loss, train/cls_loss, train/dfl_loss,
      metrics/precision(B), metrics/recall(B), metrics/mAP50(B), metrics/mAP50-95(B),
      val/box_loss, val/cls_loss, val/dfl_loss, lr/pg0..2
    第 9 列是 val/box_loss（约 2.0），按列号取会把 loss 当成 mAP 打印出来。
    """
    p = os.path.join(run_dir, 'results.csv')
    if not os.path.exists(p):
        return None
    best = None
    with open(p, encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [(x or '').strip() for x in (reader.fieldnames or [])]
        for row in reader:
            row = {(k or '').strip(): (v or '').strip() for k, v in row.items()}
            try:
                m50 = float(row['metrics/mAP50(B)'])
            except (KeyError, ValueError):
                continue
            if best is None or m50 > best:
                best = m50
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prefix', default='cv')
    ap.add_argument('--folds', default='0,1,2,3,4')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--data', default=DATA)
    ap.add_argument('--verify', action='store_true', help='只校验折划分')
    args = ap.parse_args()

    folds = [int(x) for x in args.folds.split(',') if x.strip() != '']
    ok = verify_folds(len(folds))
    if args.verify:
        raise SystemExit(0 if ok else 1)

    rows = []
    print('\n=== 逐折在同一个测试集上评测（%s）===' % args.prefix)
    for k in folds:
        run_dir = os.path.join(BASE, 'runs', '%s_f%d' % (args.prefix, k))
        w = os.path.join(run_dir, 'weights', 'best.pt')
        if not os.path.exists(w):
            print('  f%d: 跳过（没有 %s）' % (k, w))
            continue
        mb = os.path.getsize(w) / 1024 / 1024
        if mb > 10:
            print('  f%d: ⚠️ 权重 %0.1f MB（优化器未剥离）→ 这一折是中断的残局，跳过' % (k, mb))
            continue
        m = YOLO(w).val(data=args.data, split='test', imgsz=args.imgsz,
                        conf=0.001, iou=0.6, plots=False, verbose=False)
        ap50 = [None if v != v else float(v) for v in m.box.ap50]
        row = {
            'fold': k,
            'test_mAP50': float(m.box.map50),
            'test_mAP50_95': float(m.box.map),
            'per_class_ap50': dict(zip(CLASSES, ap50)),
            'val_mAP50_best': best_val_mAP(run_dir),
        }
        rows.append(row)
        print('  f%d: test mAP@0.5 = %.4f | @0.5:0.95 = %.4f | crazing AP = %s | 本折val = %s'
              % (k, row['test_mAP50'], row['test_mAP50_95'],
                 ('%.3f' % ap50[0]) if ap50[0] is not None else 'n/a',
                 ('%.4f' % row['val_mAP50_best']) if row['val_mAP50_best'] is not None else 'n/a'))

    if len(rows) < 2:
        print('\n至少需要 2 折才能算标准差。先跑 run_cv.ps1。')
        raise SystemExit(1)

    def stats(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return None
        return {
            'n': len(vals),
            'mean': st.mean(vals),
            'stdev': st.stdev(vals) if len(vals) > 1 else 0.0,
            'min': min(vals),
            'max': max(vals),
        }

    print('\n=== 汇总：噪声底（%d 折，同一个测试集）===' % len(rows))
    m50 = stats([r['test_mAP50'] for r in rows])
    m95 = stats([r['test_mAP50_95'] for r in rows])
    print('  %-22s %8s %8s %8s %8s %10s'
          % ('指标', '均值', '标准差', '最小', '最大', '2σ 参考线'))
    print('  %-22s %8.4f %8.4f %8.4f %8.4f %10.4f'
          % ('test mAP@0.5', m50['mean'], m50['stdev'], m50['min'], m50['max'], 2 * m50['stdev']))
    print('  %-22s %8.4f %8.4f %8.4f %8.4f %10.4f'
          % ('test mAP@0.5:0.95', m95['mean'], m95['stdev'], m95['min'], m95['max'], 2 * m95['stdev']))

    print('')
    print('=== 分类别 AP@0.5（跨折均值 ± 标准差）===')
    print('  %-16s %8s %8s %8s %10s' % ('class', '均值', '标准差', '最小', '2σ 参考线'))
    per_class = {}
    for c in CLASSES:
        s = stats([r['per_class_ap50'].get(c) for r in rows])
        if s is None:
            continue
        per_class[c] = s
        print('  %-16s %8.4f %8.4f %8.4f %10.4f'
              % (c, s['mean'], s['stdev'], s['min'], 2 * s['stdev']))

    print('')
    print('=== 怎么用这张表（重要）===')
    print('  1) 以后任何改动（增强/预处理/模型），都用同一套 5 折 + 同一个测试集重跑；')
    print('  2) 差异 < 2σ（当前 test mAP@0.5 的参考线 = %.4f）→ **不要声称有效**；' % (2 * m50['stdev']))
    print('  3) 差异 ≥ 2σ → 再做配对检验（同折逐折做差，看 5 个差是否同号）。')
    print('  4) 注意：这里量的是"换折"的波动。若换种子，波动通常更大，另跑一组 --prefix cv_sXX 对照。')

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, 'cv_summary_%s.json' % args.prefix)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({
            'prefix': args.prefix, 'folds': folds, 'data': args.data, 'imgsz': args.imgsz,
            'per_fold': rows,
            'test_mAP50': m50, 'test_mAP50_95': m95,
            'per_class_ap50': per_class,
            'two_sigma_line_test_mAP50': 2 * m50['stdev'],
        }, f, indent=2, ensure_ascii=False)
    print('\n  saved -> %s' % path)


if __name__ == '__main__':
    main()
