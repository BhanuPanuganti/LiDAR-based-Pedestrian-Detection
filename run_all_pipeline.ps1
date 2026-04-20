$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

Write-Host "[INFO] Project root: $projectRoot"

function Ensure-Dir([string]$Path) {
    if (-not (Test-Path $Path)) {
        New-Item -ItemType Directory -Path $Path -Force | Out-Null
    }
}

function Test-PythonImport([string]$ModuleName) {
    & python -c "import $ModuleName" 2>$null
    return ($LASTEXITCODE -eq 0)
}

function Ensure-Package([string]$ModuleName, [string]$PipName) {
    if (-not (Test-PythonImport $ModuleName)) {
        Write-Host "[INFO] Installing missing package: $PipName"
        & python -m pip install $PipName
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to install package: $PipName"
        }
    }
}

Ensure-Dir ".\outputs"
Ensure-Dir ".\outputs\results"
Ensure-Dir ".\outputs\visuals"

# Base deps used by scripts and plotting
Ensure-Package "numpy" "numpy"
Ensure-Package "pandas" "pandas"
Ensure-Package "matplotlib" "matplotlib"
Ensure-Package "tqdm" "tqdm"
Ensure-Package "yaml" "pyyaml"
Ensure-Package "easydict" "easydict"
Ensure-Package "tensorboardX" "tensorboardX"
Ensure-Package "skimage" "scikit-image"

# Optional: needed only by 3D interactive viewer
if (-not (Test-PythonImport "open3d")) {
    Write-Host "[WARN] open3d is not installed; interactive 3D viewer scripts may fail."
}

if (-not (Test-PythonImport "torch")) {
    throw "PyTorch is not available in this Python environment. Install torch first and rerun."
}

Write-Host "[STEP] Environment check"
& python -c "from setup_env import setup_environment, check_torch; setup_environment(); check_torch()"
if ($LASTEXITCODE -ne 0) { throw "Environment check failed" }

Write-Host "[STEP] Initial data run and BEV save"
& python .\main.py
if ($LASTEXITCODE -ne 0) { throw "main.py failed" }

Write-Host "[STEP] PointPillars pretrained evaluation"
& python -m deep_learning.pretrained.pointpillers.evaluate_pretrained
if ($LASTEXITCODE -ne 0) { Write-Host "[WARN] PointPillars pretrained evaluation failed" }

Write-Host "[STEP] SECOND pretrained evaluation"
& python -m deep_learning.pretrained.second.evaluate_pretrained
if ($LASTEXITCODE -ne 0) { Write-Host "[WARN] SECOND pretrained evaluation failed" }

Write-Host "[STEP] Save all visualization images"
& python .\scripts\save_all_visualizations.py
if ($LASTEXITCODE -ne 0) { throw "Saving visualizations failed" }

Write-Host "[DONE] Pipeline finished."
Write-Host "[INFO] Saved visuals:"
Get-ChildItem ".\outputs\visuals" | Sort-Object LastWriteTime -Descending | Select-Object Name, LastWriteTime

Write-Host "[INFO] Recent evaluation logs:"
Get-ChildItem ".\custom_openpcdet\output" -Recurse -Filter "log_eval_*.txt" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 10 FullName, LastWriteTime
