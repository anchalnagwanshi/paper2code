@echo off
setlocal enabledelayedexpansion

echo ==========================================
echo Paper2Code - Production Setup (Windows)
echo ==========================================
echo.

REM Check Python version
echo Checking Python version...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.10 or higher
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [OK] Python %PYTHON_VERSION%

REM Check Docker
echo.
echo Checking Docker...
docker --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker not found
    echo Please install Docker Desktop from: https://docs.docker.com/desktop/install/windows-install/
    pause
    exit /b 1
)

docker ps >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker daemon not running
    echo Please start Docker Desktop
    pause
    exit /b 1
)
echo [OK] Docker running

REM Create virtual environment
echo.
echo Creating virtual environment...
if exist venv (
    echo [WARNING] Virtual environment already exists
    set /p RECREATE="Recreate? (y/N): "
    if /i "!RECREATE!"=="y" (
        rmdir /s /q venv
        python -m venv venv
    )
) else (
    python -m venv venv
)
echo [OK] Virtual environment ready

REM Activate virtual environment
echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip setuptools wheel >nul 2>&1
echo [OK] pip upgraded

REM Install dependencies
echo.
echo Installing dependencies...
echo This may take a few minutes...
pip install -r requirements.txt >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    echo Try running manually: pip install -r requirements.txt
    pause
    exit /b 1
)
echo [OK] Dependencies installed

REM Create necessary directories
echo.
echo Creating directories...
if not exist logs mkdir logs
if not exist metrics mkdir metrics
if not exist build_context mkdir build_context
if not exist checkpoints mkdir checkpoints
if not exist .cache mkdir .cache
if not exist tests\papers mkdir tests\papers
echo [OK] Directories created

REM Check Ollama
echo.
echo Checking Ollama...
ollama --version >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Ollama not found
    echo For local LLM support, install Ollama from: https://ollama.ai
    echo Or configure to use OpenAI/Anthropic API in config.py
) else (
    echo [OK] Ollama installed
    
    REM Check if phi3 model is available
    ollama list | findstr "phi3" >nul 2>&1
    if errorlevel 1 (
        echo [WARNING] phi3 model not found
        set /p DOWNLOAD="Download phi3 model? (y/N): "
        if /i "!DOWNLOAD!"=="y" (
            ollama pull phi3
            echo [OK] phi3 model downloaded
        )
    ) else (
        echo [OK] phi3 model available
    )
)

REM Create example paper
echo.
echo Creating example paper...
(
echo Title: Image Classification with ResNet on CIFAR-10
echo.
echo Abstract:
echo We implement image classification using ResNet-50 architecture on the CIFAR-10 dataset.
echo.
echo Methodology:
echo We use a ResNet-50 convolutional neural network for image classification.
echo The model is trained on the CIFAR-10 dataset which contains 60,000 32x32 color images
echo in 10 classes.
echo.
echo Training Configuration:
echo - Model: ResNet-50
echo - Dataset: CIFAR-10 ^(from torchvision^)
echo - Optimizer: Adam
echo - Learning rate: 0.001
echo - Batch size: 32
echo - Epochs: 10
echo - Loss function: Cross-Entropy Loss
echo - GPU: Not required for this small experiment
echo.
echo Evaluation:
echo We evaluate the model on the test set using accuracy and F1-score metrics.
) > tests\papers\example_paper.txt
echo [OK] Example paper created

REM Run tests
echo.
echo Running tests...
python -m main --test
if errorlevel 1 (
    echo [WARNING] Some tests failed ^(this is OK for first setup^)
) else (
    echo [OK] All tests passed
)

REM Print summary
echo.
echo ==========================================
echo Setup Complete!
echo ==========================================
echo.
echo Next steps:
echo.
echo 1. Activate virtual environment:
echo    venv\Scripts\activate
echo.
echo 2. Try the example paper:
echo    python -m main tests\papers\example_paper.txt
echo.
echo 3. Or validate your own paper:
echo    python -m main your_paper.pdf --dry-run
echo.
echo 4. View help:
echo    python -m main --help
echo.
echo 5. View metrics:
echo    python -m main --stats
echo.
echo Documentation: README.md
echo Configuration: config.py
echo.
echo ==========================================

REM Create a quick test script
echo @echo off > quick_test.bat
echo call venv\Scripts\activate.bat >> quick_test.bat
echo python -m main tests\papers\example_paper.txt >> quick_test.bat
echo pause >> quick_test.bat

echo.
echo [OK] Created quick_test.bat for easy testing
echo Run: quick_test.bat
echo.
pause