param(
    [Parameter(Mandatory = $true)][string]$Name,
    [string]$Model = 'yolov8n.pt',
    [int]$Seed = 0,
    [int]$Imgsz = 640,
    [int]$Epochs = 50,
    [int]$Batch = 16,
    [int]$Workers = 4,
    [bool]$Cache = $true,
    [string]$Data = '',
    [string[]]$Extra = @()
)

# Usage:
#   .\scripts\train.ps1 -Name my_640
#   .\scripts\train.ps1 -Name exp_v8s -Model yolov8s.pt -Batch 8
#   .\scripts\train.ps1 -Name seed1_640 -Seed 1 -Workers 2 -Cache $false
#   .\scripts\train.ps1 -Name exp_mosaic0 -Extra mosaic=0
#   .\scripts\train.ps1 -Name exp_clahe_640 -Data E:\yolo\datasets\neu-det-yolo-clahe\data.yaml -Workers 2 -Cache $false
#
# Data root: override with env var YOLO_ROOT (default E:\yolo)
# -Data: 换数据集（例如 CLAHE 版），不传则用原始 neu-det-yolo

$ErrorActionPreference = 'Stop'

$BASE = 'E:\yolo'
if ($env:YOLO_ROOT) { $BASE = $env:YOLO_ROOT }

$yolo  = Join-Path $BASE 'venv\Scripts\yolo.exe'
$model = Join-Path $BASE ('weights\' + $Model)
if ($Data) { $data = $Data } else { $data = Join-Path $BASE 'datasets\neu-det-yolo\data.yaml' }
$proj  = Join-Path $BASE 'runs'

if (-not (Test-Path $yolo))  { throw "yolo.exe not found: $yolo" }
if (-not (Test-Path $model)) { throw "weights not found: $model" }
if (-not (Test-Path $data))  { throw "data.yaml not found: $data" }

$cliArgs = @(
    'train',
    "model=$model",
    "data=$data",
    "imgsz=$Imgsz",
    "epochs=$Epochs",
    "batch=$Batch",
    "device=0",
    "project=$proj",
    "name=$Name",
    "seed=$Seed",
    'deterministic=True',
    "cache=$Cache",
    'patience=30',
    "workers=$Workers"
)
if ($Extra.Count -gt 0) { $cliArgs += $Extra }

Write-Host "=== training ==="
Write-Host ("  base={0}" -f $BASE)
Write-Host ("  name={0} seed={1} imgsz={2} epochs={3} batch={4} workers={5} cache={6}" -f `
    $Name, $Seed, $Imgsz, $Epochs, $Batch, $Workers, $Cache)
if ($Extra.Count -gt 0) { Write-Host ("  extra: " + ($Extra -join ' ')) }
Write-Host ""

& $yolo @cliArgs
exit $LASTEXITCODE
