# 基于 YOLOv8 的钢带表面缺陷检测

用公开数据集 **NEU-DET**（东北大学热轧带钢表面缺陷库）训练目标检测模型，目标是做一个**能讲清楚、能量化、能现场演示**的工业视觉项目 —— 不是刷 SOTA。

- 数据集：NEU-DET，6 类缺陷，**实测 1770 张**（每类 294–296 张），原始标注为 VOC XML
- 模型：YOLOv8n（单目标检测，COCO 预训练迁移，3.15 M 参数 / 8.7 GFLOPs）
- 环境：Python 3.11 / torch 2.6.0+cu124 / ultralytics 8.4.183 / RTX 4060 Laptop 8 GB
- 代码在本仓库；**数据、权重、训练输出不进仓库**（体积大、可一键重建，见第 1.4 与第 7 节）

> **本仓库包含**：全部脚本（数据下载/转换/训练评测/推理）、结果指标与曲线、踩坑记录。
> **不包含**：数据集本身、模型权重（.pt）、训练输出（runs/）、演示视频 —— 均可由脚本重新生成。
> 数据集版权归原始发布方（东北大学 NEU-DET）；本仓库只含自己编写的代码与自己的实测结果。
>
> 下面命令里的 `E:\yolo` 是作者本地的示例根目录，克隆后请替换成你自己的路径。

## 三个核心数字（均在**独立测试集** 175 张 / 402 个标注框上测得）

| 指标 | 值 | 口径 |
|---|---|---|
| **mAP@0.5** | **0.8055** | 测试集，IoU=0.5（框架标准 mAP） |
| **单帧推理** | **16.5 ms**（batch=1）／**2.96 ms**（batch=16 每图） | 纯前向，CUDA events，50 次均值 |
| **框级漏检率** | **22.64%** @conf=0.25 ／ **6.97%** @conf=0.05 | 自写匹配器，IoU≥0.5，FN/402 |

> 漏检率**必须绑定置信度阈值**。把它压到 7% 的代价是误判率从 36% 升到 71% ——
> 完整权衡曲线见 4.4 节。这是产线上的成本权衡，不是一个可以单独吹的数字。

---

## 1. 数据集与预处理

### 1.1 来源
NEU-DET 是热轧带钢表面缺陷基准库，6 类：`crazing`（龟裂）、`inclusion`（夹杂）、`patches`（斑块）、`pitted_surface`（麻点）、`rolled-in_scale`（压入氧化皮）、`scratches`（划痕）。

用的是带**原始 VOC XML 标注**的公开镜像，不是别人转好的 YOLO 版本 —— 下载、审计、转换全在本仓库脚本里，任何一步都能重跑。

### 1.2 审计结果（`scripts/audit_dataset.py`）

| 项目 | 结果 |
|---|---|
| 图像数 / 标注数 | 1770 / 1770（**逐一对齐，零缺失、零不匹配**） |
| 图像尺寸 | 全部 200×200 |
| 标注框总数 | 4126 |
| 单图目标数 | 1–9 个，中位数 2 |
| 框面积（占全图） | 中位数 11.8%；<0.5% 极小框仅 1 个；>80% 整片框 115 个 |
| 退化框 / 越界框 | 0 / 0 |

**两个结论直接决定了后面所有决策：**

1. **图像级均衡 ≠ 实例级均衡**：每类图片 294–296 张几乎相等，但**框数差 2.4 倍**
   （`inclusion` 1001 个 vs `pitted_surface` 423 个）。训练时类别损失按框数加权，
   所以"类别不均衡"要看框数，不能只看图片数。
2. **图像只有 200×200**，用 `imgsz=640` 等于 **3.2 倍上采样**。因此专门做了 640 vs 320 对照实验。

### 1.3 划分（分层抽样 + 固定种子）

| split | 图片数 | 用途 |
|---|---|---|
| train | 1416 | 训练 |
| val | 179 | 每轮验证 / 选 best |
| test | 175 | **只在最后评一次，全程不参与训练和调参** |

种子 `20261021`，脚本可复现。

### 1.4 复现数据

```powershell
python scripts\fetch_dataset.py          # 下载原始 VOC 数据集（133 MB，带断点续传）
python scripts\audit_dataset.py          # 审计：类别/框数/尺寸/退化框
python scripts\convert_voc_to_yolo.py    # VOC XML -> YOLO txt + 分层划分 + data.yaml
```

---

## 2. 为什么是 YOLOv8（而不是 Faster R-CNN / 传统阈值分割）

| 方案 | 产线适配性 | 结论 |
|---|---|---|
| 传统 OpenCV（阈值/边缘/模板） | 对光照、材质批次、背景变化极敏感；缺陷形态多变时规则会爆炸 | 可作基线，作主方案不稳 |
| 两阶段（Faster R-CNN 系列） | 精度可能更高，但推理慢、部署重，产线节拍吃不消 | 不适合实时性要求 |
| **单阶段 YOLOv8** | 速度/精度平衡，生态成熟，导出 ONNX/TensorRT 方便 | **选它** |

NEU-DET 这类**缺陷与背景灰度接近、边界模糊**的数据，瓶颈不在结构创新，而在**数据质量与工作点选择**。

---

## 3. 训练配置

| 项 | baseline_640 | cmp_320 |
|---|---|---|
| 模型 | yolov8n.pt（COCO 预训练） | 同 |
| imgsz | 640 | 320 |
| epochs / patience | 50 / 30 | 50 / 30 |
| batch | 16 | 16 |
| optimizer | auto → AdamW (lr0=0.01) | 同 |
| 增强 | 默认（mosaic=1.0, fliplr=0.5, HSV 扰动） | 同 |
| seed / deterministic | 0 / True | 同 |
| cache | True（1416 张小图，可全量驻留内存） | 同 |
| 训练耗时 | **14.0 min**（16.8 s/epoch） | **11.4 min**（13.7 s/epoch） |

```powershell
yolo train model=E:\yolo\weights\yolov8n.pt data=E:\yolo\datasets\neu-det-yolo\data.yaml ^
    imgsz=640 epochs=50 batch=16 device=0 project=E:\yolo\runs name=baseline_640 ^
    seed=0 deterministic=True cache=True patience=30 workers=4
```

---

## 4. 结果

### 4.1 主结果（测试集，两种口径都给出）

| run | imgsz | best ep | mAP@0.5 | mAP@0.5:0.95 | P@0.25 | R@0.25 | 漏检@0.25 |
|---|---|---|---|---|---|---|---|
| **baseline_640** | 640 | 49 | **0.8055** | 0.4339 | 0.6386 | **0.7736** | **22.64%** |
| cmp_320 | 320 | 50 | 0.7988 | **0.4501** | **0.7060** | 0.7289 | 27.11% |

- `mAP` 来自框架标准评测（conf=0.001，全 PR 曲线积分）；
- `P/R/漏检 @0.25` 来自**自写匹配器**（`scripts/eval_pr_at_conf.py`）：conf=0.25、IoU≥0.5、
  按类别按置信度贪心匹配，详见 4.4 节为什么必须自己算。

**两种分辨率没有决定性差异**：mAP@0.5 差 0.007，mAP@0.5:0.95 反而是 320 略高。
175 张测试集 / 402 个框，这个差值在统计噪声范围内。可说的结论是：
**320 用一半算力拿到同等精度**（吞吐 1.23 vs 2.96 ms/图），640 在固定阈值下召回略高。

### 4.2 分类别（baseline_640，测试集）

| 类别 | AP@0.5 | 漏检率 @conf0.25 | 观察 |
|---|---|---|---|
| scratches（划痕） | **0.9345** | 11.11% | 长条高对比，最好检 |
| patches（斑块） | 0.9109 | 8.24% | 面积大、特征明显 |
| inclusion（夹杂） | 0.8697 | 17.02% | 点状深色，较稳 |
| pitted_surface（麻点） | 0.8516 | 24.44% | 密集小点，边界糊 |
| rolled-in_scale（氧化皮压入） | 0.6996 | 34.43% | 与背景灰度接近 |
| **crazing（龟裂）** | **0.5670** | **47.62%** | **细密网状浅纹理，最难** |

`crazing` 和 `rolled-in_scale` 是整体漏检率的主要来源（两者合计贡献了大部分 FN）。
这是**数据性质**决定的（浅纹理、低对比、无明确边界），不是训练崩了。
把指标按类别拆开，才能知道下一步该补哪类数据 —— 这也是本项目最有价值的分析结论。

### 4.3 速度（`scripts/bench_latency.py`）

| imgsz | batch=1 单帧 | batch=16 每图 | batch=16 吞吐 |
|---|---|---|---|
| 640 | 16.51 ms | **2.96 ms** | 338 img/s |
| 416 | 15.56 ms | 1.17 ms | 853 img/s |
| 320 | 17.10 ms | 1.23 ms | 813 img/s |
| 192 | 16.50 ms | 0.98 ms | 1017 img/s |

**关键发现：batch=1 时延迟在 192→640 之间几乎不变（都约 16.5 ms）。**
说明端到端单帧的瓶颈是**框架开销**（ultralytics 逐层 Python 前向 ≈100 次 kernel launch），
不是卷积算力 —— 算力差异只在批量推理时显现（640 比 320 慢 2.4 倍）。

实测的完整推理链路（`scripts/detect.py`，含解码/NMS/画框/写视频）：

| 场景 | 结果 |
|---|---|
| 175 张测试图逐张推理 | 487 个框，p50 17.9 ms/张，26.9 FPS |
| 175 帧演示视频（500×500） | 414 个框，均值 19.6 ms/帧，**50.9 FPS** |

> 要压单帧延迟，正确方向是导出 ONNX/TensorRT（去掉 Python 逐层循环），而不是继续调 imgsz。

### 4.4 工作点分析：漏检 vs 误判（`scripts/eval_pr_at_conf.py`）

产线上两类错误成本不对称：漏检 = 坏件流出；误判 = 人工复检。所以给一条曲线，不给单一数字。

| conf | TP | FP | FN | precision | recall | 漏检率 | 图像级漏检率 |
|---|---|---|---|---|---|---|---|
| 0.05 | 374 | 917 | 28 | 0.2897 | 0.9303 | **6.97%** | 14.29% |
| 0.10 | 356 | 501 | 46 | 0.4154 | 0.8856 | 11.44% | 21.71% |
| 0.15 | 339 | 330 | 63 | 0.5067 | 0.8433 | 15.67% | 29.14% |
| 0.20 | 328 | 240 | 74 | 0.5775 | 0.8159 | 18.41% | 33.71% |
| **0.25** | 311 | 176 | 91 | 0.6386 | 0.7736 | **22.64%** | 37.71% |
| 0.30 | 301 | 129 | 101 | 0.7000 | 0.7488 | 25.12% | 41.14% |
| 0.40 | 278 | 76 | 124 | 0.7853 | 0.6915 | 30.85% | 47.43% |
| 0.50 | 254 | 49 | 148 | 0.8383 | 0.6318 | 36.82% | 53.14% |
| 0.60 | 221 | 21 | 181 | 0.9132 | 0.5498 | 45.02% | 61.71% |

**怎么用这张表**：
- 要"宁可多判也不漏"（安全优先）：conf≈0.10，漏检 11.4%，代价是每 10 个报警里只有约 4 个是真缺陷；
- 要"报警即准确"（人手紧张）：conf≈0.50，precision 0.84，但漏检升到 36.8%；
- **不存在既低漏检又高精度的点** —— 想同时改善只能从模型/数据下手（见第 8 节）。

> ⚠️ **为什么必须自己算 P/R**：ultralytics 的 `val()` 返回的 `box.mp / box.mr` 是
> **PR 曲线上 F1 最优点**的值，不是你在 `conf=` 参数里指定的阈值 —— 我用 conf=0.05…0.20
> 扫了一遍，结果**完全相同**，才发现的。第一版报告因此写错了结论（"降低阈值换不来召回"），
> 自写匹配器后纠正为"降阈值确实能换召回，代价是误判"。见 `docs/实验记录.md` 第 14 条。

### 4.5 训练曲线与混淆矩阵

- 训练曲线（loss / mAP）：[`results/baseline_640_results.png`](results/baseline_640_results.png)
- 归一化混淆矩阵：[`results/baseline_640_confusion_matrix.png`](results/baseline_640_confusion_matrix.png)
- PR 曲线：[`results/baseline_640_boxPR.png`](results/baseline_640_boxPR.png)
- 320 对照：`results/cmp_320_results.png`、`results/cmp_320_confusion_matrix.png`

---

## 5. 演示

```powershell
# 单张图片
python scripts\detect.py --weights E:\yolo\runs\baseline_640\weights\best.pt ^
    --source E:\yolo\datasets\NEU-DET\IMAGES\crazing_1.jpg --save-images

# 整个测试集（导出 CSV + 标注图）
python scripts\detect.py --weights ... --source E:\yolo\datasets\neu-det-yolo\images\test ^
    --save-csv --save-images --out E:\yolo\demo\test_pred

# 视频（自动跳过首帧预热开销）
python scripts\detect.py --weights ... --source E:\yolo\demo\line_sim.mp4 --out E:\yolo\demo\video_pred
```

`scripts/make_demo_video.py` 用**测试集图像**合成 29 秒的"产线流"视频用于演示：
**不含任何训练集图像，也不是真实产线录像**。这点必须在任何场合讲清楚。

演示视频已生成：`E:\yolo\demo\video_pred\annotated.mp4`（175 帧，带类别+置信度标签）。

---

## 6. 踩坑记录（完整版见 [`docs/实验记录.md`](docs/实验记录.md)）

1. **镜像实测 1770 张，不是文献常写的 1800** —— 逐文件配对验证后接受实数，不照抄文献。
2. **PowerShell/curl 的 HTTPS 全挂（TLS 1.0 / Schannel `SEC_E_NO_CREDENTIALS`），pip 却能下 2.5 GB** —— 所有下载改走 Python 通道解决。
3. **ultralytics 8.4 会自动下 `yolo26n.pt` 做 AMP 校验**，慢网下卡死训练启动；手动预置该文件后正常。
4. **torch 下载完被杀，靠 pip 缓存 161 秒重装完成** —— 缓存放大盘是刚需。
5. **框架的 `box.mr` 是 F1 最优点，不是你的 conf 阈值** —— 导致第一版漏检率结论是错的，改自写匹配器后纠正。
6. **视频分支检测数报 0** —— 计数写在了 `if writer:`（CSV）分支里，不开 `--save-csv` 就永远为 0。检测本身没问题，是统计代码 bug。
7. **val 0.7605 vs test 0.8055** —— 175 张测试集波动大，指标必须标明在哪个集、多少样本。

---

## 7. 目录结构与本地数据

```
repo/
├─ scripts/
│  ├─ fetch_dataset.py        # 下载原始 VOC 数据集（Python 通道，断点续传）
│  ├─ fetch_file.py           # 通用断点续传下载（权重等）
│  ├─ audit_dataset.py        # 数据审计：类别/框数/尺寸/退化框
│  ├─ convert_voc_to_yolo.py  # VOC XML -> YOLO txt + 分层划分 + data.yaml
│  ├─ eval_test.py            # 框架标准 mAP + 单帧延迟
│  ├─ eval_pr_at_conf.py      # 自写匹配器：指定 conf 下的 P/R/漏检
│  ├─ bench_latency.py        # 延迟与吞吐基准（batch 1 vs 16，多 imgsz）
│  ├─ summarize_run.py        # 从 results.csv 汇总各 run
│  ├─ detect.py               # 推理程序：图片/目录/视频/摄像头 + CSV 导出
│  ├─ make_demo_video.py      # 用测试集图像合成演示流
│  └─ setup/                  # 环境搭建脚本
├─ docs/实验记录.md
└─ requirements.txt
```

本地目录（全部在 **E 盘 / NTFS**，统一一个根；数据与 venv 不进版本库）：

```
E:\yolo\
├─ venv\                          # Python 3.11 虚拟环境（torch 2.6.0+cu124）
├─ repo\                          # 本仓库（代码 + 文档）
├─ datasets\NEU-DET\              # 原始 IMAGES + ANNOTATIONS
├─ datasets\neu-det-yolo\         # 划分后的 YOLO 格式 + data.yaml
├─ runs\baseline_640\             # 权重、曲线、混淆矩阵、results.csv
├─ runs\cmp_320\
├─ runs\eval\                     # 各评测 JSON（mAP / PR@conf / 基准）
├─ weights\                       # 预训练权重（yolov8n.pt / yolo26n.pt）
├─ demo\                          # 演示视频与检测结果
└─ .pip-cache\                    # pip 缓存（重装时免下载）
```

---

## 8. 还没做的（诚实列表）

- **`crazing` 漏检 47.6%**，是最大短板；尚未做针对性处理（补数据 / 针对性增强 / 切片推理）
- **未导出 ONNX/TensorRT** —— 这是把单帧 16.5 ms 压下去的正道（去掉 Python 逐层循环），
  目前 16.5 ms 的构成已测清（框架开销 ≈ 13 ms，卷积算力 ≈ 3 ms）
- 只做了 imgsz 对照，未做 yolov8s / 注意力模块等结构对照（不是本阶段目标）
- 未做 5 折交叉验证 —— 175 张测试集本身波动大，是已知局限
- 未在真实产线视频上验证（数据集是离线图像，演示视频是合成流）

## 参考

- NEU-DET：东北大学热轧带钢表面缺陷数据库
- Ultralytics YOLOv8 官方文档
- 数据集镜像：`siddhartamukherjee/NEU-DET-Steel-Surface-Defect-Detection`（VOC 原始标注 + 原图）
