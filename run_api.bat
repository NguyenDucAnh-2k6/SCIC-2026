@echo off
setlocal enabledelayedexpansion

REM ====================================================================
REM  ClassTwin: Classroom Digital Twin (SCIC 2026)
REM  API Service & Interactive Web Dashboard Launcher
REM ====================================================================

REM Navigate to repository root directory
cd /d "%~dp0"

REM Setup PYTHONPATH
set "PYTHONPATH=%CD%;%CD%\src;%PYTHONPATH%"

REM Detect Python executable (venv or system)
set "PYTHON_CMD="
if exist ".venv\Scripts\python.exe" (
    set "PYTHON_CMD=.venv\Scripts\python.exe"
) else if exist "venv\Scripts\python.exe" (
    set "PYTHON_CMD=venv\Scripts\python.exe"
) else (
    where python >nul 2>nul
    if !errorlevel! equ 0 (
        set "PYTHON_CMD=python"
    ) else (
        where py >nul 2>nul
        if !errorlevel! equ 0 (
            set "PYTHON_CMD=py -3"
        ) else (
            echo [ERROR] Khong tim thay Python tren he thong! Vui long cai dat Python 3.10+ hoac kich hoat virtualenv.
            pause
            exit /b 1
        )
    )
)

echo ====================================================================
echo   ClassTwin: Khoi chay Digital Twin Service ^& Dashboard
echo   Python: %PYTHON_CMD%
echo   Working Dir: %CD%
echo ====================================================================

if /i "%~1"=="help" goto show_help
if /i "%~1"=="--help" goto show_help
if /i "%~1"=="-h" goto show_help

REM If no arguments provided, default to --dev (FastAPI :8000 + Vite :3000)
if "%~1"=="" (
    echo [INFO] Chay che do Development mac dinh [--dev]
    echo        FastAPI Backend : http://localhost:8000
    echo        React Dashboard : http://localhost:3000
    echo ====================================================================
    %PYTHON_CMD% scripts\run_api.py --dev
    goto end
)

echo [INFO] Chay voi tham so: %*
echo ====================================================================
%PYTHON_CMD% scripts\run_api.py %*
goto end

:show_help
echo ====================================================================
echo   ClassTwin run_api Launcher - Huong dan su dung:
echo ====================================================================
echo   run_api.bat                  : Chay mac dinh voi che do Dev (--dev)
echo   run_api.bat --dev            : Chay kem Vite dev server (port 3000)
echo   run_api.bat --port 8000      : Chi chay FastAPI backend (phuc vu frontend/dist)
echo   run_api.bat --no-browser     : Khong tu dong mo trinh duyet
echo   run_api.bat --host 0.0.0.0   : Doi dia chi host lang nghe
echo ====================================================================
:end
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Tien trinh run_api da ket thuc voi ma loi %errorlevel%.
)

endlocal
