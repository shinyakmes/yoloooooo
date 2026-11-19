# -*- coding: utf-8 -*-
"""把脚本里写死的 E:\\yolo 路径改成可用环境变量 YOLO_ROOT 覆盖（默认仍是 E:\\yolo）。

这样别人 clone 后：
    $env:YOLO_ROOT = 'D:\\my_data'   # 或 set YOLO_ROOT=...
就能直接跑，不用改代码。
"""
import io
import os
import re

SCRIPTS = r'E:\yolo\repo\scripts'

HEADER = "BASE = os.environ.get('YOLO_ROOT', r'E:\\yolo')\n"

PATCHES = {
    'fetch_dataset.py': [
        ("ROOT = r'E:\\yolo\\datasets'", "ROOT = os.path.join(BASE, 'datasets')"),
    ],
    'audit_dataset.py': [
        ("ROOT = r'E:\\yolo\\datasets\\NEU-DET'", "ROOT = os.path.join(BASE, 'datasets', 'NEU-DET')"),
        ("REPORT = r'E:\\yolo\\datasets\\audit_report.txt'",
         "REPORT = os.path.join(BASE, 'datasets', 'audit_report.txt')"),
    ],
    'convert_voc_to_yolo.py': [
        ("SRC = r'E:\\yolo\\datasets\\NEU-DET'", "SRC = os.path.join(BASE, 'datasets', 'NEU-DET')"),
        ("DST = r'E:\\yolo\\datasets\\neu-det-yolo'", "DST = os.path.join(BASE, 'datasets', 'neu-det-yolo')"),
    ],
    'eval_test.py': [
        ("DATA = r'E:\\yolo\\datasets\\neu-det-yolo\\data.yaml'",
         "DATA = os.path.join(BASE, 'datasets', 'neu-det-yolo', 'data.yaml')"),
        ("OUT_DIR = r'E:\\yolo\\runs\\eval'", "OUT_DIR = os.path.join(BASE, 'runs', 'eval')"),
    ],
    'eval_pr_at_conf.py': [
        ("LABELS = r'E:\\yolo\\datasets\\neu-det-yolo\\labels\\test'",
         "LABELS = os.path.join(BASE, 'datasets', 'neu-det-yolo', 'labels', 'test')"),
        ("IMAGES = r'E:\\yolo\\datasets\\neu-det-yolo\\images\\test'",
         "IMAGES = os.path.join(BASE, 'datasets', 'neu-det-yolo', 'images', 'test')"),
        ("OUT = r'E:\\yolo\\runs\\eval'", "OUT = os.path.join(BASE, 'runs', 'eval')"),
    ],
    'bench_latency.py': [
        ("OUT = r'E:\\yolo\\runs\\eval'", "OUT = os.path.join(BASE, 'runs', 'eval')"),
    ],
    'make_demo_video.py': [
        ("SRC = r'E:\\yolo\\datasets\\neu-det-yolo\\images\\test'",
         "SRC = os.path.join(BASE, 'datasets', 'neu-det-yolo', 'images', 'test')"),
        ("ap.add_argument('--out', default=r'E:\\yolo\\demo\\line_sim.mp4')",
         "ap.add_argument('--out', default=os.path.join(BASE, 'demo', 'line_sim.mp4'))"),
    ],
}


def main():
    for fn, pairs in PATCHES.items():
        p = os.path.join(SCRIPTS, fn)
        if not os.path.exists(p):
            print('  跳过（不存在）', fn)
            continue
        with io.open(p, encoding='utf-8') as f:
            src = f.read()
        orig = src
        # 1) 若还没有 BASE 定义，插到第一个 import os 之后
        if 'BASE = os.environ.get' not in src:
            add = '\n\n# 数据根目录：可用环境变量 YOLO_ROOT 覆盖\n' + HEADER
            if re.search(r'^import os\s*$', src, flags=re.M):
                src = re.sub(r'^(import os\s*)$', lambda m: m.group(1) + add,
                             src, count=1, flags=re.M)
            else:
                src = src.replace('import os\n', 'import os\n' + add, 1)
        # 2) 逐个替换常量
        done = 0
        for old, new in pairs:
            if old in src:
                src = src.replace(old, new)
                done += 1
            else:
                print('    ⚠️ 未找到（可能已改过）:', old[:60])
        if src != orig:
            with io.open(p, 'w', encoding='utf-8', newline='') as f:
                f.write(src)
            print('  ✅ %-26s 替换 %d/%d 处' % (fn, done, len(pairs)))
        else:
            print('  -- %-26s 无变化' % fn)


if __name__ == '__main__':
    main()
