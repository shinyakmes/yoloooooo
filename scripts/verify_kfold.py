# -*- coding: utf-8 -*-
"""核对 kfold 数据集是否真的做到了「分层 + 固定种子 + 互斥 + 不碰 test」。

检查项：
  1. 每折 val 与 train 无交集
  2. 5 折的 val 并起来 = 全部 1595 张（每张图恰好被验证一次）
  3. test 的 175 张一张都没混进任何折
  4. 每个类在每折 val 里的数量是否接近 1/5（分层的意义）

用法：
  python verify_kfold.py
"""
import json
import os

CLASS_PREFIX = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
SRC = os.environ.get('YOLO_ROOT', r'E:\yolo')
MAIN = os.path.join(SRC, 'datasets', 'neu-det-yolo')
KF = os.path.join(SRC, 'datasets', 'kfold')
K = 5


def stems(path):
    return {os.path.splitext(f)[0] for f in os.listdir(path) if f.lower().endswith('.jpg')}


def cls_of(stem):
    return stem.rsplit('_', 1)[0]


def main():
    test = stems(os.path.join(MAIN, 'images', 'test'))
    pool = stems(os.path.join(MAIN, 'images', 'train')) | stems(os.path.join(MAIN, 'images', 'val'))
    print('测试集 %d 张 / 交叉验证池(train+val) %d 张 / 合计 %d' % (len(test), len(pool), len(test) + len(pool)))

    fold_val, fold_train = {}, {}
    ok = True
    for k in range(K):
        d = os.path.join(KF, 'f%d' % k)
        va = stems(os.path.join(d, 'images', 'val'))
        tr = stems(os.path.join(d, 'images', 'train'))
        fold_val[k], fold_train[k] = va, tr
        leak = (va | tr) & test
        inter = va & tr
        print('\n--- f%d ---' % k)
        print('  train=%d  val=%d  val∩train=%d  test 泄漏=%d' % (len(tr), len(va), len(inter), len(leak)))
        if inter:
            ok = False
            print('  [X] val 与 train 有重叠，示例：', sorted(inter)[:5])
        if leak:
            ok = False
            print('  [X] 测试集图片混进折里，示例：', sorted(leak)[:5])
        if va | tr != pool:
            ok = False
            print('  [X] 该折覆盖的图片不等于池子: 缺 %d, 多 %d'
                  % (len(pool - (va | tr)), len((va | tr) - pool)))

    union = set()
    for k in range(K):
        union |= fold_val[k]
    print('\n5 折 val 并集 = %d （应等于 %d）' % (len(union), len(pool)))
    if union != pool:
        ok = False
        print('  [X] 并集不等于池子: 缺 %d' % len(pool - union))
    # 每张图恰好被验证一次
    cnt = {}
    for k in range(K):
        for s in fold_val[k]:
            cnt[s] = cnt.get(s, 0) + 1
    dup = [s for s, c in cnt.items() if c != 1]
    if dup:
        ok = False
        print('  [X] 有 %d 张图被验证次数 != 1，示例：%s' % (len(dup), dup[:5]))
    else:
        print('  [OK] 每张图恰好被验证一次（互斥且完备）')

    # 分层检查
    print('\n=== 分层检查：每折 val 里的各类数量（整池的 1/5 作为理想值）===')
    total = {c: sum(1 for s in pool if cls_of(s) == c) for c in CLASS_PREFIX}
    print('  %-16s %7s %7s' % ('class', '池内', '理想/折'))
    for c in CLASS_PREFIX:
        print('  %-16s %7d %7.1f' % (c, total[c], total[c] / K))
    print('')
    print('  %-16s %s' % ('class', ''.join('f%-6d' % k for k in range(K))))
    worst = 0.0
    for c in CLASS_PREFIX:
        row = []
        for k in range(K):
            n = sum(1 for s in fold_val[k] if cls_of(s) == c)
            row.append(n)
            ideal = total[c] / K
            worst = max(worst, abs(n - ideal) / ideal if ideal else 0)
        print('  %-16s %s' % (c, ''.join('%-7d' % n for n in row)))
    print('\n  各类在各折间相对理想值的最大偏差 = %.1f%%' % (100 * worst))
    if worst > 0.15:
        print('  !️  偏差偏大（>15%），分层可能有问题')
    else:
        print('  [OK] 各折分布均衡（偏差 <15%）')

    out = {'test': len(test), 'pool': len(pool),
           'fold_val': {k: len(v) for k, v in fold_val.items()},
           'per_class_total': total,
           'max_rel_dev': worst, 'ok': ok}
    p = os.path.join(SRC, 'runs', 'eval', 'kfold_verify.json')
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print('\n  saved -> %s' % p)
    print('\n结论：%s' % ('[OK] 划分通过全部检查' if ok else '[X] 划分有问题，先修再训练'))


if __name__ == '__main__':
    main()
