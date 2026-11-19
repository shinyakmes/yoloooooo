# -*- coding: utf-8 -*-
"""对比 PT 与 ONNX 的原始输出，定位 ONNX 精度差异的来源。

用法：python compare_onnx_pt.py [--imgsz 640]
"""
import argparse
import os

import cv2
import numpy as np
import onnxruntime as ort
import torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox

W = r'E:\yolo\runs\baseline_640\weights'
PT = os.path.join(W, 'best.pt')
ONNX = os.path.join(W, 'best.onnx')
IMG = r'E:\yolo\datasets\neu-det-yolo\images\test\crazing_1.jpg'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--imgsz', type=int, default=640)
    a = ap.parse_args()

    # ---- ONNX 元数据（class names 是否带进去了）----
    sess = ort.InferenceSession(ONNX, providers=['CPUExecutionProvider'])
    meta = sess.get_modelmeta()
    print('=== ONNX metadata ===')
    print('  producer:', getattr(meta, 'producer_name', '?'), getattr(meta, 'producer_version', '?'))
    cm = meta.custom_metadata_map
    print('  custom keys:', list(cm.keys()))
    for k in ('names', 'imgsz', 'stride', 'task', 'head', 'end2end', 'batch', 'args'):
        if k in cm:
            v = str(cm[k])
            print('  %-8s = %s' % (k, v[:200]))

    # ---- 准备同一份输入 ----
    img = cv2.imread(IMG)
    lb = LetterBox(a.imgsz, auto=False, stride=32)
    im = lb(image=img)
    x = im[None].transpose(0, 3, 1, 2)[..., ::-1].astype(np.float32) / 255.0
    xt = torch.from_numpy(x).contiguous()
    print('\n=== input ===')
    print('  numpy shape', x.shape, 'range', float(x.min()), float(x.max()))

    # ---- PyTorch 原始输出 ----
    model = YOLO(PT)
    net = model.model.float().eval()
    with torch.no_grad():
        out = net(xt)
    print('\n=== PyTorch raw output type ===')
    print('  type:', type(out))
    if isinstance(out, dict):
        print('  keys:', list(out.keys()))
        for k, v in out.items():
            if isinstance(v, torch.Tensor):
                print('    %s: shape %s' % (k, tuple(v.shape)))
            elif isinstance(v, (list, tuple)):
                print('    %s: list of %d' % (k, len(v)))
    if isinstance(out, (list, tuple)):
        # 每个尺度 (1, 4+nc, H, W) -> (1, 4+nc, H*W)
        parts = [o.reshape(o.shape[0], o.shape[1], -1) for o in out]
        pt = torch.cat(parts, dim=2)
    elif isinstance(out, dict):
        cand = None
        for k, v in out.items():
            if isinstance(v, torch.Tensor) and v.dim() == 3 and v.shape[1] >= 5:
                cand = v
                break
        if cand is None:
            print('  ⚠️ 没有找到可对比的 3D 张量；直接用第一个 tensor 试试')
            cand = next(v for v in out.values() if isinstance(v, torch.Tensor))
        pt = cand
    else:
        pt = out
    pt_np = pt.numpy()
    print('\n=== PyTorch raw output ===')
    print('  shape', pt_np.shape, 'dtype', pt_np.dtype)

    # ---- ONNX 原始输出 ----
    onnx_out = sess.run(None, {sess.get_inputs()[0].name: x})
    print('\n=== ONNX raw output ===')
    for i, o in enumerate(onnx_out):
        print('  out[%d] shape=%s dtype=%s' % (i, o.shape, o.dtype))
    onx = onnx_out[0]

    # ---- 对比 ----
    print('\n=== compare ===')
    if pt_np.shape == onx.shape:
        d = np.abs(pt_np - onx)
        print('  max |diff| = %.6g' % d.max())
        print('  mean|diff| = %.6g' % d.mean())
        print('  PT  score range: %.4f ~ %.4f' % (pt_np[:, 4:, :].min(), pt_np[:, 4:, :].max()))
        print('  ONX score range: %.4f ~ %.4f' % (onx[:, 4:, :].min(), onx[:, 4:, :].max()))
        print('  PT  box  range: %.2f ~ %.2f' % (pt_np[:, :4, :].min(), pt_np[:, :4, :].max()))
        print('  ONX box  range: %.2f ~ %.2f' % (onx[:, :4, :].min(), onx[:, :4, :].max()))
    else:
        print('  ⚠️ 形状不同：PT', pt_np.shape, 'vs ONNX', onx.shape)


if __name__ == '__main__':
    main()
