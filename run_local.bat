@echo off
echo ========================================================
echo  Starting Deepfake Detection Web Application Locally
echo ========================================================

cd /d "%~dp0"

IF NOT EXIST ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment .venv not found!
    echo Please create the virtual environment first using:
    echo python -m venv .venv
    pause
    exit /b 1
)

IF NOT EXIST "Django Application\models" (
    echo [INFO] Linking models directory...
    cmd /c mklink /J "Django Application\models" "models"
)

echo [INFO] Activating virtual environment...
call .venv\Scripts\activate.bat

echo [INFO] Starting Django development server at http://127.0.0.1:8000/
cd "Django Application"
python manage.py runserver 127.0.0.1:8000
pause
