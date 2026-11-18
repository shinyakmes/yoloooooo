param(
    [Parameter(Mandatory = $true)][string]$Name,
    [int]$Seed = 0,
    [int]$Imgsz = 640,
    [int]$Epochs = 50,
    [int]$Batch = 16,
    [int]$Workers = 4,
    [bool]$Cache = $true,
    [string[]]$Extra = @()
)

# Usage:
#   .\scripts\train.ps1 -Name my_run
#   .\scripts\train.ps1 -Name seed1_640 -Seed 1
#   .\scripts\train.ps1 -Name exp_mosaic0 -Extra mosaic=0
#   .\scripts\train.ps1 -Name quick_test -Epochs 3 -Imgsz 320

$ErrorActionPreference = 'Stop'

$yolo  = 'E:\yolo\venv\Scripts\yolo.exe'
$model = 'E:\yolo\weights\yolov8n.pt'
$data  = 'E:\yolo\datasets\neu-det-yolo\data.yaml'

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
    'project=E:\yolo\runs',
    "name=$Name",
    "seed=$Seed",
    'deterministic=True',
    "cache=$Cache",
    'patience=30',
    "workers=$Workers"
)
if ($Extra.Count -gt 0) { $cliArgs += $Extra }

Write-Host "=== training ==="
Write-Host ("  name={0} seed={1} imgsz={2} epochs={3} batch={4} workers={5}" -f $Name, $Seed, $Imgsz, $Epochs, $Batch, $Workers)
if ($Extra.Count -gt 0) { Write-Host ("  extra: " + ($Extra -join ' ')) }
Write-Host ""

& $yolo @cliArgs
exit $LASTEXITCODE
