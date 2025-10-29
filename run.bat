@echo off
REM Quick start script for BETAFPV Configurator - UltraThink Edition (Windows)

echo ================================================
echo BETAFPV Configurator - UltraThink Edition
echo ================================================
echo.

REM Check if conda is available
where conda >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo Checking for Anaconda environment...
    conda env list | find "betafpv-ultrathink" >nul 2>nul
    if %ERRORLEVEL% EQU 0 (
        echo Environment found. Activating...
        call conda activate betafpv-ultrathink
    ) else (
        echo Creating Anaconda environment...
        call conda env create -f environment.yml
        call conda activate betafpv-ultrathink
    )
) else (
    echo Anaconda not found. Using system Python...
    if not exist "venv" (
        echo Creating virtual environment...
        python -m venv venv
    )
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
    echo Installing dependencies...
    pip install -r requirements.txt
)

echo.
echo Launching BETAFPV Configurator...
echo.
python main.py
