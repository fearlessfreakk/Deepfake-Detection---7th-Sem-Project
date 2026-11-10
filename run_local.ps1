Write-Host "========================================================" -ForegroundColor Cyan
Write-Host " Starting Deepfake Detection Web Application Locally" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

$venvPython = Join-Path $scriptDir ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Error "[ERROR] Virtual environment .venv not found at $venvPython!"
    exit 1
}

$appModels = Join-Path $scriptDir "Django Application\models"
if (-not (Test-Path $appModels)) {
    Write-Host "[INFO] Linking models directory..." -ForegroundColor Yellow
    cmd /c mklink /J "Django Application\models" "models"
}

Write-Host "[INFO] Starting Django development server at http://127.0.0.1:8000/" -ForegroundColor Green
Set-Location (Join-Path $scriptDir "Django Application")
& $venvPython manage.py runserver 127.0.0.1:8000
