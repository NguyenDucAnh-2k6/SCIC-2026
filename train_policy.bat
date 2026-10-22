@echo off
setlocal enabledelayedexpansion

REM ====================================================================
REM  ClassTwin: Pedagogical Policy Training (CQL & Inverse RL)
REM ====================================================================

cd /d "%~dp0"
set "PYTHONPATH=%CD%;%CD%\src;%PYTHONPATH%"

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
            echo [ERROR] Khong tim thay Python tren he thong!
            exit /b 1
        )
    )
)

if "%~1"=="" (
    echo [INFO] Huan luyen CQL Policy mac dinh [Explicit Reward, 40 episodes, 150 steps]...
    %PYTHON_CMD% scripts\train_policy.py --reward-mode explicit --episodes 40 --train-steps 150
) else (
    %PYTHON_CMD% scripts\train_policy.py %*
)

endlocal
