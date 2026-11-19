# -*- coding: utf-8 -*-
"""生成 5 折交叉验证的数据目录与 data yaml。

思路（为什么不简单地把 val 拆小）：
  现在的 val 只有 179 张，噪声底 0.0064。5 折的目的是**让每一次评估都用 1/5 的数据做验证，
  且每张图都被验证过一次**，从而得到"均值 ± 标准差"，而不是单次训练的一次偶然结果。

  注意：test 175 张完全不动，仍然只在最后用一次。

产出：
  E:\\yolo\\datasets\\kfold\\
    f0\\images\\train  f0\\images\\val  f0\\labels\\train  f0\\labels\\val
    ...
    data_f0.yaml ... data_f4.yaml
"""
import os
import random
import shutil
from collections import defaultdict

SRC = r'E:\yolo\datasets\neu-det-yolo'
DST = r'E:\yolo\datasets\kfold'
CLASSES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
K = 5
SEED = 20261119


def main():
    # 1) 收集 train + val 的图片（不含 test）
    stems = []
    for split in ('train', 'val'):
        d = os.path.join(SRC, 'images', split)
        for f in os.listdir(d):
            if f.lower().endswith('.jpg'):
                stems.append(os.path.splitext(f)[0])
    print('可用图片（train+val，不含 test）=', len(stems))

    # 2) 按类别分层分折
    by_cls = defaultdict(list)
    for s in stems:
        by_cls[s.rsplit('_', 1)[0]].append(s)
    rng = random.Random(SEED)
    fold_of = {}
    for cls, lst in by_cls.items():
        lst = sorted(lst)
        rng.shuffle(lst)
        for i, s in enumerate(lst):
            fold_of[s] = i % K
    print('每折图片数 =', [sum(1 for s in stems if fold_of[s] == k) for k in range(K)])

    # 3) 建目录并复制（图片小，直接复制最省事，避免软链接的跨盘问题）
    for k in range(K):
        for sub in ('images/train', 'images/val', 'labels/train', 'labels/val'):
            os.makedirs(os.path.join(DST, 'f%d' % k, sub), exist_ok=True)

    n_copy = 0
    for s in stems:
        vf = fold_of[s]
        img_src = None
        for split in ('train', 'val'):
            p = os.path.join(SRC, 'images', split, s + '.jpg')
            if os.path.exists(p):
                img_src = p
                break
        lbl_src = None
        for split in ('train', 'val'):
            p = os.path.join(SRC, 'labels', split, s + '.txt')
            if os.path.exists(p):
                lbl_src = p
                break
        if not img_src or not lbl_src:
            print('  跳过（缺文件）', s)
            continue
        for k in range(K):
            role = 'val' if k == vf else 'train'
            shutil.copy2(img_src, os.path.join(DST, 'f%d' % k, 'images', role, s + '.jpg'))
            shutil.copy2(lbl_src, os.path.join(DST, 'f%d' % k, 'labels', role, s + '.txt'))
            n_copy += 2

    # 4) 写 data yaml（不含绝对路径，跟主数据集保持一致的做法）
    for k in range(K):
        y = os.path.join(DST, 'data_f%d.yaml' % k)
        with open(y, 'w', encoding='utf-8') as f:
            f.write('# kfold f%d（自动生成，test 不参与）\n' % k)
            f.write('train: f%d/images/train\n' % k)
            f.write('val: f%d/images/val\n' % k)
            f.write('\nnames:\n')
            for i, c in enumerate(CLASSES):
                f.write('  %d: %s\n' % (i, c))
        print('  ->', y)

    print('复制文件数 =', n_copy)
    for k in range(K):
        tr = len(os.listdir(os.path.join(DST, 'f%d' % k, 'images/train')))
        va = len(os.listdir(os.path.join(DST, 'f%d' % k, 'images/val')))
        print('  f%d: train=%d  val=%d' % (k, tr, va))


if __name__ == '__main__':
    main()
