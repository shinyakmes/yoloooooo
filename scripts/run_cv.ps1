# ============================================================
#  交叉验证协议 v2 · 跑批脚本
# ============================================================
#  为什么重做（旧 kfold 的问题）：
#    - reads f2/f3 的权重是 31.8 MB（优化器未剥离）→ 中途被杀的残局
#    - f4 从来没跑过
#    - 只有 f0/f1 是干净的 2 折，报不出标准差
#
#  本脚本的规矩：
#    1. 5 折全部用同一份配置（yolov8n / 640 / 50 轮 / seed 0 / workers 2 / cache False）
#    2. 折的划分由 make_kfold.py 固定（seed 20261119），本脚本不重新划分
#    3. **每折模型最后都要在同一个 175 张测试集上评一次**（见 eval_cv_test.py）
#       —— 这样得到的 5 个数才是"配对"的，才能做配对检验
#    4. 可断点续跑：已经跑完（权重已剥离）的折自动跳过
#
#  用法：
#    powershell -ExecutionPolicy Bypass -File E:\yolo\repo\scripts\run_cv.ps1
#    ... -Folds 0,1                 # 只跑指定折
#    ... -Prefix cv_s42 -Seed 42    # 换个种子，用于估计"种子噪声"
# ============================================================

param(
    [string]$Prefix = 'cv',
    [int]$Epochs = 50,
    [int]$Batch = 16,
    [int]$Workers = 2,
    [int]$Seed = 0,
    [int]$Imgsz = 640,
    [int[]]$Folds = @(0, 1, 2, 3, 4),
    [switch]$Force
)

$ErrorActionPreference = 'Continue'
$yolo   = 'E:\yolo\venv\Scripts\yolo.exe'
$model  = 'E:\yolo\weights\yolov8n.pt'
$root   = 'E:\yolo\datasets\kfold'
$runs   = 'E:\yolo\runs'
$logs   = Join-Path $runs 'cv_logs'
New-Item -ItemType Directory -Force -Path $logs | Out-Null

if (-not (Test-Path $yolo))  { throw "找不到 $yolo" }
if (-not (Test-Path $model)) { throw "找不到 $model" }

Write-Host "================================================================"
Write-Host " 交叉验证协议 v2    prefix=$Prefix  seed=$Seed  imgsz=$Imgsz"
Write-Host " epochs=$Epochs batch=$Batch workers=$Workers   folds=$($Folds -join ',')"
Write-Host " 开始时间 $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "================================================================"

$done = @(); $skipped = @(); $failed = @()

foreach ($k in $Folds) {
    $data = Join-Path $root ("data_f{0}.yaml" -f $k)
    $name = "{0}_f{1}" -f $Prefix, $k
    $runDir = Join-Path $runs $name
    $best   = Join-Path $runDir 'weights\best.pt'
    $log    = Join-Path $logs ("{0}_f{1}.log" -f $Prefix, $k)

    if (-not (Test-Path $data)) { Write-Host "  [skip] 缺少数据集 $data"; continue }

    # 断点续跑：权重存在且已剥离优化器（< 10 MB）视为跑完
    if ((-not $Force) -and (Test-Path $best)) {
        $mb = (Get-Item $best).Length / 1MB
        if ($mb -lt 10) {
            Write-Host ("  [done] {0} 已存在且已剥离（{1:N1} MB），跳过" -f $name, $mb)
            $skipped += $k
            continue
        } else {
            Write-Host ("  [redo] {0} 权重 {1:N1} MB = 优化器未剥离 = 上次被中断，重跑" -f $name, $mb)
        }
    }

    Write-Host ""
    Write-Host ("--- fold {0}  开始 {1} ---" -f $k, (Get-Date -Format 'HH:mm:ss'))
    $sw = [System.Diagnostics.Stopwatch]::StartNew()

    # 全部输出进日志，避免控制台缓冲问题（实验记录第 10 条）
    & $yolo train model=$model data=$data imgsz=$Imgsz epochs=$Epochs batch=$Batch `
        device=0 project=$runs name=$name seed=$Seed deterministic=True `
        cache=False patience=30 workers=$Workers *>&1 |
        Tee-Object -FilePath $log | Out-Null

    $code = $LASTEXITCODE
    $sw.Stop()

    $ok = $false
    if (Test-Path $best) {
        $mb = (Get-Item $best).Length / 1MB
        # 跑完的标志：出现 "Optimizer stripped" 且权重回到正常大小
        $stripped = (Select-String -Path $log -Pattern 'Optimizer stripped' -Quiet)
        $ok = ($mb -lt 10) -and $stripped
        Write-Host ("    fold {0} 用时 {1:N1} 分钟 | exit={2} | best.pt={3:N1} MB | stripped={4} | 判定={5}" `
            -f $k, $sw.Elapsed.TotalMinutes, $code, $mb, $stripped, $(if ($ok) { 'OK' } else { '异常' }))
    } else {
        Write-Host ("    fold {0} 用时 {1:N1} 分钟 | exit={2} | 没有产出 best.pt → 失败" `
            -f $k, $sw.Elapsed.TotalMinutes, $code)
    }

    if ($ok) { $done += $k } else { $failed += $k }
}

Write-Host ""
Write-Host "================================================================"
Write-Host (" 本次跑完 : {0}" -f $(if ($done.Count) { $done -join ',' } else { '无' }))
Write-Host (" 跳过(已有): {0}" -f $(if ($skipped.Count) { $skipped -join ',' } else { '无' }))
Write-Host (" 失败/异常 : {0}" -f $(if ($failed.Count) { $failed -join ',' } else { '无' }))
Write-Host " 结束时间 $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "================================================================"
Write-Host ""
Write-Host " 下一步（每折都在同一个测试集上评一次）："
Write-Host ("  E:\yolo\venv\Scripts\python.exe E:\yolo\repo\scripts\eval_cv_test.py --prefix {0}" -f $Prefix)

if ($failed.Count -gt 0) { exit 1 }
exit 0
