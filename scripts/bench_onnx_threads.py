# -*- coding: utf-8 -*-
"""ONNX Runtime CPU 线程数对延迟的影响（batch=1，纯前向）。

对比参考：PyTorch CPU 前向约 67 ms/图（来自 yolo val 的 Speed 行）。
"""
import time

import numpy as np
import onnxruntime as ort

ONNX = r'E:\yolo\runs\baseline_640\weights\best.onnx'
IMGSZ = 640
ITERS = 30
WARMUP = 5


def bench(threads):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(ONNX, sess_options=so, providers=['CPUExecutionProvider'])
    name = sess.get_inputs()[0].name
    x = np.random.rand(1, 3, IMGSZ, IMGSZ).astype(np.float32)
    for _ in range(WARMUP):
        sess.run(None, {name: x})
    t0 = time.perf_counter()
    for _ in range(ITERS):
        sess.run(None, {name: x})
    ms = (time.perf_counter() - t0) / ITERS * 1000
    return ms


if __name__ == '__main__':
    print('%-10s %10s' % ('线程数', 'ms/图'))
    for t in (1, 2, 4, 6, 8, 0):   # 0 = ORT 自动
        ms = bench(t)
        print('%-10s %10.1f' % (t if t else 'auto', ms))
