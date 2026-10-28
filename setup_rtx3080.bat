@echo off
chcp 65001 >nul
title GFPGAN Setup - RTX 3080 (CUDA 12.1)
echo ============================================================
echo  GFPGAN Setup - RTX 3080 (Ampere, CUDA 12.1)
echo ============================================================
echo.

:: ---- Check Python ----
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Install Python 3.10 and add to PATH.
    echo Download: https://www.python.org/downloads/release/python-31011/
    pause
    exit /b 1
)
python --version 2>&1 | findstr /r "3\.10 3\.11 3\.12" >nul
if %errorlevel% neq 0 (
    echo [WARN] Recommend Python 3.10-3.12. Current:
    python --version
    echo Continue anyway...
)

:: ---- Check NVIDIA GPU ----
where nvidia-smi >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] nvidia-smi not found. Install NVIDIA driver first.
    echo Download: https://www.nvidia.com/download/index.aspx
    pause
    exit /b 1
)
echo [GPU Info]
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo.

:: ---- Create venv ----
if exist .venv (
    echo [INFO] .venv already exists, skipping creation.
) else (
    echo [1/5] Creating virtual environment...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create venv.
        pause
        exit /b 1
    )
)

:: ---- Upgrade pip ----
echo [2/5] Upgrading pip...
.venv\Scripts\python.exe -m pip install --upgrade pip >nul 2>&1

:: ---- Install PyTorch cu121 ----
echo [3/5] Installing PyTorch (CUDA 12.1 for RTX 3080)...
.venv\Scripts\pip.exe install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install PyTorch.
    pause
    exit /b 1
)

:: ---- Install GFPGAN dependencies ----
echo [3/5] Installing GFPGAN dependencies...
.venv\Scripts\pip.exe install basicsr==1.4.2 facexlib==0.3.0 realesrgan==0.3.0 gfpgan==1.3.8 opencv-python numpy scipy pyyaml tqdm lmdb yapf
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

:: ---- Patch basicsr degradations.py ----
echo [4/5] Patching basicsr (functional_tensor fix)...
set "PATCH_FILE=.venv\lib\site-packages\basicsr\data\degradations.py"
if exist "%PATCH_FILE%" (
    powershell -Command "(Get-Content '%PATCH_FILE%') -replace 'from torchvision\.transforms\.functional_tensor import rgb_to_grayscale','from torchvision.transforms.functional import rgb_to_grayscale' | Set-Content '%PATCH_FILE%' -Encoding utf8"
    echo [OK] Patched %PATCH_FILE%
) else (
    echo [WARN] %PATCH_FILE% not found, skipping patch.
)

:: ---- Download model weights ----
echo [5/5] Downloading model weights...

if not exist "experiments\pretrained_models" mkdir "experiments\pretrained_models"
if not exist "gfpgan\weights" mkdir "gfpgan\weights"

if not exist "experiments\pretrained_models\GFPGANv1.4.pth" (
    echo Downloading GFPGANv1.4.pth (332MB)...
    powershell -Command "Invoke-WebRequest -Uri 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth' -OutFile 'experiments\pretrained_models\GFPGANv1.4.pth'"
    if %errorlevel% neq 0 (
        echo [WARN] Auto-download failed. Download manually:
        echo   https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth
        echo   Save to: experiments\pretrained_models\GFPGANv1.4.pth
    ) else (
        echo [OK] GFPGANv1.4.pth downloaded.
    )
) else (
    echo [OK] GFPGANv1.4.pth already exists.
)

if not exist "gfpgan\weights\detection_Resnet50_Final.pth" (
    echo Downloading detection_Resnet50_Final.pth (104MB)...
    powershell -Command "Invoke-WebRequest -Uri 'https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth' -OutFile 'gfpgan\weights\detection_Resnet50_Final.pth'"
) else (
    echo [OK] detection_Resnet50_Final.pth already exists.
)

if not exist "gfpgan\weights\parsing_parsenet.pth" (
    echo Downloading parsing_parsenet.pth (81MB)...
    powershell -Command "Invoke-WebRequest -Uri 'https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth' -OutFile 'gfpgan\weights\parsing_parsenet.pth'"
) else (
    echo [OK] parsing_parsenet.pth already exists.
)

:: ---- Verify ----
echo.
echo ============================================================
echo  Verifying installation...
echo ============================================================
.venv\Scripts\python.exe -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo.
echo ============================================================
echo  Setup complete!
echo ============================================================
echo.
echo  Test static image:
echo    .venv\Scripts\activate
echo    python inference_gfpgan.py -i inputs/whole_imgs -o results -v 1.4 --bg_upsampler none
echo.
echo  Test real-time IP camera:
echo    python realtime_gfpgan.py
echo    (Edit IP_CAMERA_URL in realtime_gfpgan.py first)
echo.
pause
