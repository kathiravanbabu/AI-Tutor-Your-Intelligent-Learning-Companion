@echo off
setlocal

echo ==========================================
echo   AI Tutor: Starting Application...
echo ==========================================

REM Check for virtual environment
if not exist "venv\Scripts\activate.bat" (
    echo [INFO] Virtual environment not found. Creating one...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo [INFO] Installing dependencies...
    pip install -r requirements.txt
) else (
    echo [INFO] Activating virtual environment...
    call venv\Scripts\activate.bat
)

echo [INFO] launching Streamlit...
streamlit run app.py

pause
