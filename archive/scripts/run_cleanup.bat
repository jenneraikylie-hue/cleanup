@echo off
REM Vinyl Playmat Digital Restoration - Quick Launcher
REM This batch file launches the HSV-based restoration script (recommended)

echo ============================================
echo Vinyl Playmat Restoration Tool (HSV Mode)
echo ============================================
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

REM Run the HSV-based restoration script
echo Starting HSV-based image restoration...
echo Processing all images in current directory...
echo Output will be saved to "output" folder
echo.
echo Using HSV color space for robust detection
echo (For legacy BGR version, run: python restore_playmat.py)
echo.

python restore_playmat_hsv.py .

if errorlevel 1 (
    echo.
    echo ERROR: Script encountered an error
    pause
    exit /b 1
)

echo.
echo ============================================
echo Processing Complete!
echo ============================================
echo Check the "output" folder for cleaned images
echo.
pause
