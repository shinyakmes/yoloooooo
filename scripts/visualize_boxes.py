# -*- coding: utf-8 -*-
"""把标注框和预测框亲手画到图上（不用框架的 plot 函数）。

这个脚本存在的理由：
  视频里那种"跑通 YOLO、看到框"的操作，框架一句 results.plot() 就替你做完了，
  但你不知道框为什么会出现在那个位置。而工业部署里，框最终要送到上位机 / PLC /
  界面，坐标必须由你自己算。

所以这里把整条坐标链拆开，每一步都打印出来：
  1. 标注文件里的数            (cx, cy, w, h)  全部是 0~1 的归一化值
  2. 乘以原图宽高              -> 原图像素坐标
  3. letterbox 缩放 + 补边     -> 640x640 网络输入下的坐标
  4. 网络输出 (1, 4+nc, 8400)  -> 仍然是 640 坐标系、中心点格式
  5. 反 letterbox (减 pad, 除 r) -> 回到原图坐标系
  6. OpenCV 画框                -> 你看到的那个框

用法：
  # 只画标注（不需要模型，最安全，训练期间也能跑）
  python visualize_boxes.py --image E:\\yolo\\datasets\\neu-det-yolo\\images\\test\\crazing_1.jpg --mode gt

  # 标注 + 预测对照（需要 ONNX，走 CPU，不抢 GPU）
  python visualize_boxes.py --image <同上> --onnx E:\\yolo\\repo\\results\\best.onnx --mode both

输出：
  E:\\yolo\\runs\\viz\\<图名>_gt.png      只有标注框（绿）
  E:\\yolo\\runs\\viz\\<图名>_pred.png    只有预测框（红）
  E:\\yolo\\runs\\viz\\<图名>_both.png    两个叠一起（绿=标注，红=预测）
"""
import argparse
import os
import sys

import cv2
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE = os.environ.get('YOLO_ROOT', r'E:\yolo')
LABELS = os.path.join(BASE, 'datasets', 'neu-det-yolo', 'labels', 'test')
OUT = os.path.join(BASE, 'runs', 'viz')
CLASSES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
COLOR_GT = (0, 200, 0)      # 绿 BGR
COLOR_PRED = (0, 0, 220)    # 红


# ---------------------------------------------------------------- 标注部分
def load_gt(stem, w, h, verbose=True):
    """读 YOLO txt -> [(cls, x1, y1, x2, y2)]，同时把归一化->像素的换算打印出来。"""
    p = os.path.join(LABELS, stem + '.txt')
    if not os.path.exists(p):
        return []
    out = []
    rows = [ln.split() for ln in open(p, encoding='utf-8') if len(ln.split()) >= 5]
    if verbose:
        print('\n=== 1. 标注文件 %s ===' % os.path.basename(p))
        print('    格式: 类别 cx cy w h  —— 后四个都是 0~1 的归一化值，不是像素！')
        print('    原图尺寸: %d x %d' % (w, h))
    for i, r in enumerate(rows):
        c = int(float(r[0]))
        cx, cy, bw, bh = (float(v) for v in r[1:5])
        x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
        x2, y2 = (cx + bw / 2) * w, (cy + bh / 2) * h
        out.append((c, x1, y1, x2, y2))
        if verbose and i < 3:
            print('    [%d] cls=%d(%s) 归一化(%.3f,%.3f,%.3f,%.3f) -> 像素(%.1f,%.1f)-(%.1f,%.1f)'
                  % (i, c, CLASSES[c], cx, cy, bw, bh, x1, y1, x2, y2))
        if verbose and i == 3:
            print('    ... 共 %d 个框' % len(rows))
    return out


# ---------------------------------------------------------------- 预处理
def letterbox(img, new=640):
    """等比缩放 + 居中补边。返回 图, r(缩放比), (dw, dh)(每边补的宽高)。"""
    h, w = img.shape[:2]
    r = min(new / w, new / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    dw, dh = (new - nw) / 2, (new - nh) / 2
    if (nw, nh) != (new, new):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img, r, (dw, dh)


# ---------------------------------------------------------------- 推理（ONNX CPU）
def predict_onnx(onnx_path, img_bgr, conf=0.25, iou=0.5, verbose=True):
    import onnxruntime as ort

    h, w = img_bgr.shape[:2]
    lb, r, (dw, dh) = letterbox(img_bgr, 640)
    if verbose:
        print('\n=== 2. letterbox（网络只吃 640x640）===')
        print('    原图 %dx%d -> 缩放比 r=%.4f -> %dx%d，左右补 %.1f / 上下补 %.1f 像素'
              % (w, h, r, int(round(w * r)), int(round(h * r)), dw, dh))
        if abs(dw) < 1e-6 and abs(dh) < 1e-6:
            print('    注意：本数据集是 200x200 正方形、目标是 640 正方形，所以 r=3.2 且不补边。')
            print('          换成产线 1920x1080 的图，这一步就会出现补边，框最容易在这里画偏。')

    inp = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    inp = inp.transpose(2, 0, 1)[None]  # (1,3,640,640)
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    name = sess.get_inputs()[0].name
    out = sess.run(None, {name: inp})[0]  # (1, 4+nc, 8400)
    if verbose:
        print('\n=== 3. 网络输出 ===')
        print('    shape = %s   -> (1, 4+类别数, 候选框数)' % (out.shape,))
        print('    %d 个候选 = 80x80 + 40x40 + 20x20 = %d+%d+%d（三个尺度的网格）'
              % (out.shape[2], 6400, 1600, 400))
        print('    每个候选 = cx,cy,w,h + %d 个类别分数，坐标单位是 640 输入图像素'
              % (out.shape[1] - 4))

    pred = out[0].T  # (8400, 4+nc)
    boxes_xywh = pred[:, :4]
    scores_all = pred[:, 4:]
    cls_ids = scores_all.argmax(1)
    scores = scores_all.max(1)
    keep = scores > conf
    boxes_xywh, cls_ids, scores = boxes_xywh[keep], cls_ids[keep], scores[keep]
    if verbose:
        print('    conf>%.2f 之后剩 %d 个候选（这一步在部署里叫"打分过滤"）' % (conf, len(scores)))

    # xywh(中心点) -> xyxy(角点)
    xyxy = np.empty_like(boxes_xywh)
    xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2

    # 逐类 NMS（同一个物体只留一个框）
    keep_idx = []
    for c in np.unique(cls_ids):
        idx = np.where(cls_ids == c)[0]
        rects = [[float(xyxy[i, 0]), float(xyxy[i, 1]),
                  float(xyxy[i, 2] - xyxy[i, 0]), float(xyxy[i, 3] - xyxy[i, 1])] for i in idx]
        k = cv2.dnn.NMSBoxes(rects, scores[idx].tolist(), conf, iou)
        keep_idx.extend(idx[np.array(k).ravel()] if len(k) else [])
    keep_idx = np.array(sorted(keep_idx), dtype=int)
    if verbose:
        print('    NMS(iou=%.2f) 之后剩 %d 个（同一个物体只留一个框）' % (iou, len(keep_idx)))
    xyxy, cls_ids, scores = xyxy[keep_idx], cls_ids[keep_idx], scores[keep_idx]

    # 反 letterbox：先减补边，再除缩放比
    xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - dw) / r
    xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - dh) / r
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, w - 1)
    xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, h - 1)
    if verbose:
        print('\n=== 4. 反 letterbox 回到原图坐标 ===')
        print('    x = (x_640 - %.2f) / %.4f    y = (y_640 - %.2f) / %.4f' % (dw, r, dh, r))
        print('    这是画框最容易错的一步：只除 r 不减 pad，图越大框偏得越多。')
    return [(int(c), float(s), *[float(v) for v in b]) for c, s, b in zip(cls_ids, scores, xyxy)]


# ---------------------------------------------------------------- 画图
def draw(img, boxes, color, with_score=False, thick=1):
    im = img.copy()
    for b in boxes:
        if with_score:
            c, s, x1, y1, x2, y2 = b[0], b[1], b[2], b[3], b[4], b[5]
            label = '%s %.2f' % (CLASSES[c], s)
        else:
            c, x1, y1, x2, y2 = b[0], b[1], b[2], b[3], b[4]
            label = CLASSES[c]
        p1, p2 = (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2)))
        cv2.rectangle(im, p1, p2, color, thick)
        ty = max(p1[1] - 3, 8)
        cv2.putText(im, label, (p1[0], ty), cv2.FONT_HERSHEY_SIMPLEX, 0.33, color, 1, cv2.LINE_AA)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True)
    ap.add_argument('--onnx', default='')
    ap.add_argument('--mode', choices=['gt', 'pred', 'both'], default='gt')
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--iou', type=float, default=0.5)
    ap.add_argument('--scale', type=float, default=2.5, help='输出放大倍数（原图只有 200x200）')
    a = ap.parse_args()

    img = cv2.imread(a.image)
    if img is None:
        raise SystemExit('读不到图: %s' % a.image)
    h, w = img.shape[:2]
    stem = os.path.splitext(os.path.basename(a.image))[0]
    os.makedirs(OUT, exist_ok=True)
    big = lambda im: cv2.resize(im, (int(w * a.scale), int(h * a.scale)),
                                interpolation=cv2.INTER_NEAREST)

    gt = load_gt(stem, w, h)
    print('\n标注框 %d 个' % len(gt))
    if a.mode in ('gt', 'both'):
        p = os.path.join(OUT, stem + '_gt.png')
        cv2.imwrite(p, big(draw(img, gt, COLOR_GT)))
        print('  saved -> %s' % p)

    if a.mode in ('pred', 'both'):
        if not a.onnx:
            raise SystemExit('--mode %s 需要 --onnx' % a.mode)
        preds = predict_onnx(a.onnx, img, a.conf, a.iou)
        print('\n预测框 %d 个:' % len(preds))
        for c, s, x1, y1, x2, y2 in preds:
            print('  %-15s %.3f  (%.1f,%.1f)-(%.1f,%.1f)' % (CLASSES[c], s, x1, y1, x2, y2))
        p = os.path.join(OUT, stem + '_pred.png')
        cv2.imwrite(p, big(draw(img, preds, COLOR_PRED, with_score=True)))
        print('  saved -> %s' % p)

    if a.mode == 'both':
        both = draw(img, gt, COLOR_GT)
        both = draw(both, [(c,) + tuple(b[1:]) for c, s, *b in preds], COLOR_PRED, thick=1)
        p = os.path.join(OUT, stem + '_both.png')
        cv2.imwrite(p, big(both))
        print('  saved -> %s   (绿=标注, 红=预测)' % p)


if __name__ == '__main__':
    main()
