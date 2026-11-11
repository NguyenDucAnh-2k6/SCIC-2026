@echo off
setlocal enabledelayedexpansion

REM ====================================================================
REM  ClassTwin: Classroom Digital Twin (SCIC 2026)
REM  Training Launcher (Single Master Pipeline: scripts\pipeline.py)
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

REM Check CLI arguments
if not "%~1"=="" (
    if /i "%~1"=="help" goto show_help
    if /i "%~1"=="--help" goto show_help
    if /i "%~1"=="-h" goto show_help

    if /i "%~1"=="smoke" goto run_smoke
    if /i "%~1"=="fast" goto run_smoke
    if /i "%~1"=="--smoke" goto run_smoke
    if /i "%~1"=="--fast" goto run_smoke

    if /i "%~1"=="preprocess" goto run_preprocess
    if /i "%~1"=="estimator" goto run_estimator
    if /i "%~1"=="dkt" goto run_estimator

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
echo   [1] Full Pipeline    - Toan bo chu trinh (Preprocess + DKT + CQL + Eval)
echo   [2] Fast Smoke Test  - Kiem thu nhanh CPU (2 shards, 1 epoch, ~10s)
echo   [3] Preprocess Only  - Trich xuat Parquet shards ^& priors 7 bo du lieu
echo   [4] Estimator Only   - Huan luyen mo hinh GRU-DKT
echo   [5] Policy: CQL      - Huan luyen Chinh sach CQL (Explicit Reward)
echo   [6] Policy: MaxEnt   - Huan luyen Chinh sach MaxEnt Inverse RL (IRL)
echo   [7] Evaluate Only    - Chay benchmark danh gia Calibration ^& Entropy
echo   [0] Thoat
echo ====================================================================
set /p "CHOICE=Lua chon cua ban [1-7, mac dinh 1]: "

if "%CHOICE%"=="" set "CHOICE=1"
if "%CHOICE%"=="1" goto run_full
if "%CHOICE%"=="2" goto run_smoke
if "%CHOICE%"=="3" goto run_preprocess
if "%CHOICE%"=="4" goto run_estimator
if "%CHOICE%"=="5" goto run_policy_explicit
if "%CHOICE%"=="6" goto run_policy_irl
if "%CHOICE%"=="7" goto run_evaluate
if "%CHOICE%"=="0" goto end

echo [WARN] Lua chon khong hop le, chay mac dinh Full Pipeline...
goto run_full

:run_full
echo.
echo [INFO] Bat dau Full Pipeline (Smart auto-detection shards)...
%PYTHON_CMD% scripts\pipeline.py --stage all
goto end

:run_smoke
echo.
echo [INFO] Bat dau Fast Smoke Test (~10s)...
%PYTHON_CMD% scripts\pipeline.py --stage all --num-shards 2 --max-students 50 --estimator-epochs 1 --policy-episodes 5 --policy-steps 10
goto end

:run_preprocess
echo.
echo [INFO] Bat dau Tien xu ly da nguon (7 Datasets)...
%PYTHON_CMD% scripts\pipeline.py --stage preprocess
goto end

:run_estimator
echo.
echo [INFO] Bat dau Huan luyen GRU-DKT State Estimator...
%PYTHON_CMD% scripts\pipeline.py --stage estimator
goto end

:run_policy_explicit
echo.
echo [INFO] Bat dau Huan luyen Chinh sach CQL (Explicit Multi-Objective Reward)...
%PYTHON_CMD% scripts\pipeline.py --stage policy --reward-mode explicit --policy-episodes 40 --policy-steps 150
goto end

:run_policy_irl
echo.
echo [INFO] Bat dau Huan luyen Chinh sach MaxEnt Inverse RL (IRL)...
%PYTHON_CMD% scripts\pipeline.py --stage policy --reward-mode irl --policy-episodes 40 --policy-steps 150
goto end

:run_evaluate
echo.
echo [INFO] Bat dau Danh gia Benchmark Models ^& Policy Entropy...
%PYTHON_CMD% scripts\pipeline.py --stage evaluate
goto end

:show_help
echo ====================================================================
echo   ClassTwin Training Launcher - Huong dan su dung:
echo ====================================================================
echo   train.bat                  : Hien menu lua chon tuong tac
echo   train.bat smoke / fast     : Chay kiem thu nhanh tren CPU (~10 giay)
echo   train.bat preprocess       : Chi tien xu ly va trich xuat priors 7 datasets
echo   train.bat estimator        : Chi huan luyen GRU-DKT state estimator
echo   train.bat policy           : Huan luyen chinh sach CQL (Explicit)
echo   train.bat irl              : Huan luyen chinh sach MaxEnt Inverse RL
echo   train.bat eval             : Chay benchmark danh gia mo hinh
echo   train.bat [pipeline args]  : Truyen tham so truc tiep vao pipeline.py
echo.
echo   Vi du:
echo     train.bat --stage policy --reward-mode explicit
echo     train.bat --num-shards 5 --estimator-epochs 3
echo ====================================================================
goto end

:end
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Tien trinh da ket thuc voi ma loi %errorlevel%.
)
endlocal
