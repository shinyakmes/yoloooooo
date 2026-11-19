# -*- coding: utf-8 -*-
"""汇总 5 折交叉验证结果，输出 均值 ± 标准差（并生成可直接贴进 README 的表格）。

用法：
  python summarize_kfold.py
  python summarize_kfold.py --runs E:\\yolo\\runs          # 指定 runs 根目录
"""
import argparse
import glob
import os
import statistics as st

import csv


def best_of(csv_path):
    """返回该 run 的 (best mAP50, best mAP50-95, 对应 epoch)"""
    with open(csv_path, newline='', encoding='utf-8', errors='replace') as fh:
        rows = [{k.strip(): (v or '').strip() for k, v in r.items()} for r in csv.DictReader(fh)]
    if not rows:
        return None
    best = None
    for r in rows:
        try:
            m50 = float(r['metrics/mAP50(B)'])
            m95 = float(r['metrics/mAP50-95(B)'])
            ep = int(float(r['epoch']))
        except (KeyError, ValueError):
            continue
        if best is None or m50 > best[0]:
            best = (m50, m95, ep)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default=r'E:\yolo\runs')
    ap.add_argument('--pattern', default='kfold_f*')
    a = ap.parse_args()

    runs = sorted(glob.glob(os.path.join(a.runs, a.pattern)))
    print('=== 5 折交叉验证汇总 ===')
    if not runs:
        print('  没有找到 %s \\ %s' % (a.runs, a.pattern))
        print('  先跑：powershell -ExecutionPolicy Bypass -File E:\\yolo\\repo\\scripts\\run_kfold.ps1')
        return

    rows = []
    for d in runs:
        csv_path = os.path.join(d, 'results.csv')
        if not os.path.exists(csv_path):
            print('  跳过（无 results.csv）', os.path.basename(d))
            continue
        b = best_of(csv_path)
        if b is None:
            print('  跳过（results.csv 为空）', os.path.basename(d))
            continue
        rows.append((os.path.basename(d), b[0], b[1], b[2]))

    if not rows:
        print('  没有可用的折。')
        return

    print('')
    print('| 折 | best mAP@0.5 | best mAP@0.5:0.95 | 最佳轮次 |')
    print('|---|---|---|---|')
    for name, m50, m95, ep in rows:
        print('| %s | %.4f | %.4f | %d |' % (name, m50, m95, ep))

    m50s = [r[1] for r in rows]
    m95s = [r[2] for r in rows]
    print('')
    print('--- 统计 ---')
    print('  折数            = %d' % len(rows))
    print('  mAP@0.5         = %.4f ± %.4f   (min %.4f / max %.4f)'
          % (st.mean(m50s), st.pstdev(m50s) if len(m50s) > 1 else 0, min(m50s), max(m50s)))
    print('  mAP@0.5:0.95    = %.4f ± %.4f   (min %.4f / max %.4f)'
          % (st.mean(m95s), st.pstdev(m95s) if len(m95s) > 1 else 0, min(m95s), max(m95s)))
    if len(m50s) > 1:
        print('  → 两倍标准差（判断"改动是否有效"的参考线）= %.4f' % (2 * st.pstdev(m50s)))

    print('')
    print('可直接贴进 README 的一行：')
    print('  5 折交叉验证：mAP@0.5 = **%.4f ± %.4f**（%d 折，每折用 1/5 数据做验证；test 未参与）'
          % (st.mean(m50s), st.pstdev(m50s) if len(m50s) > 1 else 0, len(rows)))


if __name__ == '__main__':
    main()
