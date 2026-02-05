@echo off
REM ============================================================================
REM VINYL PLAYMAT RESTORATION - START HERE
REM ============================================================================
REM This is the ONLY file you need to run!
REM
REM USAGE:
REM   1. Put your scanned images in the "scans" folder
REM   2. Double-click this START_HERE.bat file
REM   3. Wait for processing to complete
REM   4. Find cleaned images in "scans/output" folder
REM ============================================================================

title Vinyl Playmat Restoration - HSV Color Detection
color 0A
echo.
echo ============================================================================
echo     VINYL PLAYMAT RESTORATION - HSV COLOR DETECTION
echo ============================================================================
echo.
echo This script will:
echo   - Check Python installation
echo   - Install required packages (opencv-python, numpy)
echo   - Process all images in the "scans" folder with optimized settings:
echo     * Preserve original green outlines (no outline normalization)
echo     * Skip slow spec removal for faster processing
echo     * Use natural green detection from original image
echo     * Preserve holes in text (no infilling)
echo     * Enable GPU acceleration if available (install cupy-cuda12x for GPU)
echo   - Output cleaned images to "scans/output"
echo.
echo ============================================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo.
    echo Please install Python 3.7 or higher from:
    echo https://www.python.org/downloads/
    echo.
    echo Make sure to check "Add Python to PATH" during installation
    echo.
    pause
    exit /b 1
)

echo [OK] Python found:
python --version
echo.

REM Check if scans folder exists
if not exist "scans" (
    echo [WARNING] "scans" folder not found - creating it now...
    mkdir scans
    echo.
    echo [INFO] Please put your scanned images in the "scans" folder
    echo        Then run this batch file again.
    echo.
    pause
    exit /b 0
)

REM Check if there are images in scans folder
dir /b "scans\*.jpg" "scans\*.jpeg" "scans\*.png" >nul 2>&1
if errorlevel 1 (
    echo [WARNING] No images found in "scans" folder
    echo.
    echo Please put your scanned JPG or PNG images in the "scans" folder
    echo Then run this batch file again.
    echo.
    pause
    exit /b 0
)

echo [OK] Images found in "scans" folder
echo.

REM Install required packages
echo ============================================================================
echo Installing required Python packages...
echo ============================================================================
echo.

python -m pip install --quiet opencv-python numpy
if errorlevel 1 (
    echo [ERROR] Failed to install required packages
    echo.
    pause
    exit /b 1
)

echo [OK] Packages installed successfully
echo.

REM Run the restoration script with optimized options
echo ============================================================================
echo Processing images with HSV color detection...
echo ============================================================================
echo.
echo Using optimized settings:
echo   - Preserving original green outlines (--skip-outline-normalization)
echo   - Skipping slow spec removal (--skip-despec)
echo   - Using natural green detection (--use-natural-green)
echo   - Preserving holes in text (--skip-infill)
echo   - GPU acceleration enabled if available (--use-gpu)
echo.

python restore_playmat_hsv.py scans/ --skip-outline-normalization --skip-despec --use-natural-green --skip-infill --use-gpu
if errorlevel 1 (
    echo.
    echo [ERROR] Processing failed - check error messages above
    echo.
    pause
    exit /b 1
)

echo.
echo ============================================================================
echo     PROCESSING COMPLETE!
echo ============================================================================
echo.
echo Your cleaned images are in: scans\output\
echo.
echo You can now:
echo   - View the cleaned images
echo   - Process more images by adding them to "scans" folder
echo   - Run this batch file again to process new images
echo.
echo ============================================================================
echo.
pause
