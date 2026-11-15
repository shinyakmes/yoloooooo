$ErrorActionPreference = 'Continue'
$ProgressPreference = 'SilentlyContinue'
$env:PIP_DISABLE_PIP_VERSION_CHECK = '1'
$env:PIP_CACHE_DIR = 'E:\yolo\.pip-cache'
$log = 'E:\yolo\setup_venv_E.log'

function W([string]$m) {
    $line = (Get-Date -Format 'HH:mm:ss') + '  ' + $m
    Write-Host $line
    Add-Content -Path $log -Value $line
}

$sw = [System.Diagnostics.Stopwatch]::StartNew()
W '=== rebuild venv on E: ==='

W 'step1: create venv'
& py -3.11 -m venv 'E:\yolo\venv'
W ('  exit=' + $LASTEXITCODE)

$py = 'E:\yolo\venv\Scripts\python.exe'
if (-not (Test-Path $py)) { W 'FATAL: venv missing'; exit 2 }

W 'step2: upgrade pip (needed for resumable downloads)'
& $py -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
W ('  exit=' + $LASTEXITCODE)

W 'step3: torch + torchvision cu124 (from cache)'
& $py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
W ('  exit=' + $LASTEXITCODE)

W 'step4: ultralytics + opencv + matplotlib + numpy (from cache)'
& $py -m pip install ultralytics opencv-python matplotlib numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
W ('  exit=' + $LASTEXITCODE)

W 'step5: verify'
& $py -c "import torch;print('torch', torch.__version__);print('cuda', torch.cuda.is_available());print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
& $py -c "import ultralytics,cv2,numpy;print('ultralytics', ultralytics.__version__);print('cv2', cv2.__version__)"
& py -3.11 -c "print('base py ok')"

W ('=== DONE total=' + [int]$sw.Elapsed.TotalSeconds + 's ===')
