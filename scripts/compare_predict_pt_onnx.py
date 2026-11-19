# -*- coding: utf-8 -*-
"""同一张图，PT 与 ONNX 的 predict 结果是否一致（单图、batch=1）。"""
import numpy as np
from ultralytics import YOLO

IMG = r'E:\yolo\datasets\neu-det-yolo\images\test\crazing_1.jpg'
PT = r'E:\yolo\runs\baseline_640\weights\best.pt'
ONNX = r'E:\yolo\runs\baseline_640\weights\best.onnx'


def boxes(model_path, device):
    m = YOLO(model_path)
    r = m.predict(IMG, imgsz=640, conf=0.001, iou=0.6, device=device, verbose=False, max_det=300)[0]
    return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()


def report(tag, b, c, k):
    print('  %-12s 框数=%2d  最高分=%.4f  平均分=%.4f  类别=%s'
          % (tag, len(b), c.max() if len(c) else 0, c.mean() if len(c) else 0, sorted(set(k.tolist()))[:6]))


if __name__ == '__main__':
    bp, cp, kp = boxes(PT, 0)
    bo, co, ko = boxes(ONNX, 'cpu')
    print('=== 单图 predict 对比（conf=0.001, iou=0.6）===')
    report('PyTorch-GPU', bp, cp, kp)
    report('ONNX-CPU', bo, co, ko)

    # 逐框比对：PT 的每个框，在 ONNX 里找 IoU 最大的框
    def iou_matrix(a, b):
        if len(a) == 0 or len(b) == 0:
            return np.zeros((len(a), len(b)))
        ax1, ay1, ax2, ay2 = a[:, 0][:, None], a[:, 1][:, None], a[:, 2][:, None], a[:, 3][:, None]
        bx1, by1, bx2, by2 = b[:, 0][None], b[:, 1][None], b[:, 2][None], b[:, 3][None]
        iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
        ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
        inter = iw * ih
        ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
        return np.where(ua > 0, inter / np.maximum(ua, 1e-9), 0)

    M = iou_matrix(bp, bo)
    if M.size:
        best = M.max(axis=1)
        print('\n  PT 每个框在 ONNX 里的最大 IoU: 最小 %.4f  中位 %.4f  最大 %.4f'
              % (best.min(), np.median(best), best.max()))
        print('  IoU>0.99 的框数: %d / %d' % (int((best > 0.99).sum()), len(best)))
        print('  分数最大差: %.6f' % np.abs(np.sort(cp)[::-1][:min(len(cp), len(co))]
                                            - np.sort(co)[::-1][:min(len(cp), len(co))]).max()
              if len(cp) and len(co) else '  (有一边没有框)')
