# 5 折交叉验证 · 跑批脚本
# 用法（挂机用，约 50 分钟）：
#   powershell -ExecutionPolicy Bypass -File E:\yolo\repo\scripts\run_kfold.ps1
# 想先只跑一折验证链路：
#   powershell -ExecutionPolicy Bypass -File E:\yolo\repo\scripts\run_kfold.ps1 -Only 0

param(
    [int]$K = 5,
    [int]$Epochs = 50,
    [int]$Batch = 16,
    [int]$Workers = 2,
    [int[]]$Only = @()
)

$ErrorActionPreference = 'Continue'
$yolo  = 'E:\yolo\venv\Scripts\yolo.exe'
$model = 'E:\yolo\weights\yolov8n.pt'
$root  = 'E:\yolo\datasets\kfold'
$logs  = 'E:\yolo\runs\kfold_logs'
New-Item -ItemType Directory -Force -Path $logs | Out-Null

$folds = if ($Only.Count -gt 0) { $Only } else { 0..($K - 1) }
Write-Host "=== k-fold 开始：folds = $($folds -join ',')  epochs=$Epochs  workers=$Workers ==="

foreach ($k in $folds) {
    $data = Join-Path $root ("data_f{0}.yaml" -f $k)
    if (-not (Test-Path $data)) { Write-Host "  [skip] 缺少 $data"; continue }
    $log = Join-Path $logs ("f{0}.log" -f $k)
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    Write-Host ""
    Write-Host ("--- fold {0}  ({1}) ---" -f $k, (Get-Date -Format 'HH:mm:ss'))

    & $yolo train model=$model data=$data imgsz=640 epochs=$Epochs batch=$Batch `
        device=0 project=E:\yolo\runs name=("kfold_f{0}" -f $k) seed=0 `
        deterministic=True cache=False patience=30 workers=$Workers 2>&1 |
        Tee-Object -FilePath $log | Select-String -Pattern 'Starting training|epochs completed|Results saved|Error|Traceback' |
        ForEach-Object { $_.Line }

    Write-Host ("    fold {0} 用时 {1:N1} 分钟  (exit={2})" -f $k, $sw.Elapsed.TotalMinutes, $LASTEXITCODE)
}

Write-Host ""
Write-Host "=== 全部结束，下一步跑汇总： ==="
Write-Host "  E:\yolo\venv\Scripts\python.exe E:\yolo\repo\scripts\summarize_kfold.py"
