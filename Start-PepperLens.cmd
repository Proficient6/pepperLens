@echo off
setlocal
cd /d "%~dp0"
echo PepperLens - Bell pepper leaf DSP lab
echo.
if not exist "website\.venv\Scripts\python.exe" (
  echo Creating the website Python environment...
  py -3.12 -m venv "website\.venv"
  if errorlevel 1 goto failed
)
"website\.venv\Scripts\python.exe" -X utf8 -c "import cv2,mahotas,matplotlib,numpy,pandas,PIL,joblib,sklearn; assert sklearn.__version__=='1.6.1'; assert numpy.__version__=='2.1.3'" >nul 2>&1
if errorlevel 1 (
  echo Installing the dependencies that match your saved model...
  "website\.venv\Scripts\python.exe" -X utf8 -m pip install -r "website\requirements-tested.txt"
  if errorlevel 1 goto failed
)
echo Starting http://127.0.0.1:7860
echo Keep this window open while using the website. Ctrl+C stops it.
"website\.venv\Scripts\python.exe" -X utf8 "website\server.py" --open-browser
if errorlevel 1 goto failed
exit /b 0
:failed
echo.
echo PepperLens could not start. Read the error above.
echo See README.md for manual setup and port instructions.
pause
exit /b 1
