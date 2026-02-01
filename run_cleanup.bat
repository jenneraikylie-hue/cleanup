@echo off
REM Vinyl Playmat Digital Restoration - Quick Launcher
REM This batch file launches the Python restoration script

echo ===================================
echo Vinyl Playmat Restoration Tool
echo ===================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.7 or later from https://www.python.org/
    echo.
    pause
    exit /b 1
)

REM Check if OpenCV is installed
python -c "import cv2" >nul 2>&1
if errorlevel 1 (
    echo Installing required dependencies...
    echo Installing opencv-python and numpy...
    echo.
    python -m pip install opencv-python numpy
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies
        echo Please run: pip install opencv-python numpy
        echo.
        pause
        exit /b 1
    )
    echo.
    echo Dependencies installed successfully!
    echo.
)

REM Run the restoration script
echo Starting image restoration...
echo Processing all JPG images in current directory...
echo Output will be saved to "restored" folder
echo.

python restore_playmat.py

if errorlevel 1 (
    echo.
    echo ERROR: Script encountered an error
    pause
    exit /b 1
)

echo.
echo ===================================
echo Processing Complete!
echo ===================================
echo Check the "restored" folder for output files
echo.
pause
