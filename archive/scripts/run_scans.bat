@echo off
REM ============================================================
REM Batch file to run HSV playmat restoration on scans folder
REM ============================================================

echo.
echo ============================================================
echo    Vinyl Playmat Restoration - HSV Implementation
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.7+ from https://www.python.org
    echo.
    pause
    exit /b 1
)

echo Python found: 
python --version
echo.

REM Check for required packages
echo Checking for required packages...
python -c "import cv2, numpy" >nul 2>&1
if errorlevel 1 (
    echo.
    echo Installing required packages (opencv-python, numpy)...
    echo This may take a few minutes...
    pip install opencv-python numpy
    if errorlevel 1 (
        echo.
        echo ERROR: Failed to install required packages
        echo Please run: pip install opencv-python numpy
        echo.
        pause
        exit /b 1
    )
    echo.
    echo Packages installed successfully!
)

echo Required packages are installed.
echo.

REM Check if scans folder exists
if not exist "scans" (
    echo ERROR: 'scans' folder not found in current directory
    echo.
    echo Please create a 'scans' folder and place your images there.
    echo Current directory: %CD%
    echo.
    pause
    exit /b 1
)

REM Check if there are images in scans folder
dir /b scans\*.jpg scans\*.jpeg scans\*.png >nul 2>&1
if errorlevel 1 (
    echo ERROR: No image files found in 'scans' folder
    echo.
    echo Please add .jpg, .jpeg, or .png files to the scans folder.
    echo.
    pause
    exit /b 1
)

echo Found images in scans folder
echo.

REM Run the HSV script
echo Starting restoration process...
echo Output will be saved to: scans\output\
echo.
echo ============================================================
echo.

python restore_playmat_hsv.py scans/

REM Check if script ran successfully
if errorlevel 1 (
    echo.
    echo ============================================================
    echo ERROR: Script execution failed
    echo ============================================================
    echo.
) else (
    echo.
    echo ============================================================
    echo Restoration complete!
    echo ============================================================
    echo.
    echo Output images saved to: scans\output\
    echo.
)

echo.
pause
