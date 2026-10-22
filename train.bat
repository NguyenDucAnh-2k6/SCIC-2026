@echo off
setlocal enabledelayedexpansion

REM ====================================================================
REM  ClassTwin: Classroom Digital Twin (SCIC 2026)
REM  Training Launcher (Full Pipeline & Policy Training)
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

REM Check if arguments were passed
if not "%~1"=="" (
    if /i "%~1"=="help" goto show_help
    if /i "%~1"=="--help" goto show_help
    if /i "%~1"=="-h" goto show_help

    if /i "%~1"=="smoke" goto run_smoke
    if /i "%~1"=="fast" goto run_smoke
    if /i "%~1"=="--smoke" goto run_smoke
    if /i "%~1"=="--fast" goto run_smoke

    if /i "%~1"=="policy" goto run_policy_explicit
    if /i "%~1"=="cql" goto run_policy_explicit
    if /i "%~1"=="--policy" goto run_policy_explicit

    if /i "%~1"=="irl" goto run_policy_irl
    if /i "%~1"=="policy-irl" goto run_policy_irl
    if /i "%~1"=="--irl" goto run_policy_irl

    if /i "%~1"=="eval" goto run_evaluate
    if /i "%~1"=="evaluate" goto run_evaluate

    REM Forward all arguments directly to scripts\pipeline.py
    echo [INFO] Chay pipeline voi tham so: %*
    %PYTHON_CMD% scripts\pipeline.py %*
    goto end
)

:interactive_menu
echo ====================================================================
echo          ClassTwin: Classroom Digital Twin (SCIC 2026)
echo                  Training Execution Launcher
echo ====================================================================
echo   [1] Full Pipeline    - Toan bo chu trinh (DKT + CQL + Manifest)
echo   [2] Fast Smoke Test  - Kiem thu nhanh CPU (2 shards, 1 epoch, ~10s)
echo   [3] Policy: CQL      - Huan luyen lai CQL (Explicit Reward)
echo   [4] Policy: MaxEnt   - Huan luyen lai Inverse RL (Latent Reward)
echo   [5] Evaluate Models  - Danh gia Entropy, Calibration ^& Policy
echo   [0] Thoat
echo ====================================================================
set /p "CHOICE=Lua chon cua ban [1-5, mac dinh 1]: "

if "%CHOICE%"=="" set "CHOICE=1"
if "%CHOICE%"=="1" goto run_full
if "%CHOICE%"=="2" goto run_smoke
if "%CHOICE%"=="3" goto run_policy_explicit
if "%CHOICE%"=="4" goto run_policy_irl
if "%CHOICE%"=="5" goto run_evaluate
if "%CHOICE%"=="0" goto end

echo [WARN] Lua chon khong hop le, chay mac dinh Full Pipeline...
goto run_full

:run_full
echo.
echo [INFO] Bat dau Full Pipeline (Smart auto-detection shards)...
%PYTHON_CMD% scripts\pipeline.py
goto end

:run_smoke
echo.
echo [INFO] Bat dau Fast Smoke Test (~10s)...
%PYTHON_CMD% scripts\pipeline.py --num-shards 2 --max-students 50 --estimator-epochs 1 --policy-episodes 5 --policy-steps 10
goto end

:run_policy_explicit
echo.
echo [INFO] Bat dau Huan luyen Chinh sach CQL (Explicit Multi-Objective Reward)...
%PYTHON_CMD% scripts\train_policy.py --reward-mode explicit --episodes 40 --train-steps 150
goto end

:run_policy_irl
echo.
echo [INFO] Bat dau Huan luyen Chinh sach MaxEnt Inverse RL (IRL)...
%PYTHON_CMD% scripts\train_policy.py --reward-mode irl --episodes 40 --train-steps 150
goto end

:run_evaluate
echo.
echo [INFO] Bat dau Danh gia Models ^& Policy Entropy...
%PYTHON_CMD% scripts\evaluate_models.py
goto end

:show_help
echo ====================================================================
echo   ClassTwin Training Launcher - Huong dan su dung:
echo ====================================================================
echo   train.bat                  : Hien menu lua chon tuong tac
echo   train.bat smoke / fast     : Chay kiem thu nhanh tren CPU (~10 giay)
echo   train.bat policy           : Huan luyen chinh sach CQL (Explicit)
echo   train.bat irl              : Huan luyen chinh sach MaxEnt Inverse RL
echo   train.bat eval             : Chay benchmark danh gia mo hinh
echo   train.bat [pipeline args]  : Truyen tham so truc tiep vao pipeline.py
echo.
echo   Vi du:
echo     train.bat --num-shards 5 --estimator-epochs 3
echo     train.bat --skip-preprocess --policy-episodes 50
echo ====================================================================
goto end

:end
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Tien trinh da ket thuc voi ma loi %errorlevel%.
)
endlocal
