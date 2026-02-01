# Vinyl Playmat Restoration

Restores scanned vinyl playmats to perfectly flat-color PNGs by removing texture, wrinkles, and scanner artifacts. Uses HSV color space for accurate color detection under varying lighting conditions.

## Quick Start (3 Steps)

### ⭐ Windows Users
1. **Put your scanned images in the `scans/` folder**
2. **Double-click `START_HERE.bat`**
3. **Find cleaned images in `scans/output/`**

### Command Line
```bash
# Install dependencies
pip install opencv-python numpy

# Process images
python restore_playmat_hsv.py scans/
```

That's it! The START_HERE.bat script automatically installs dependencies and processes all images.

## What It Does

- **Removes**: Vinyl wrinkles, plastic texture, specular highlights, scanner artifacts, marbling
- **Preserves**: Logos, text, stars, silhouettes, outlines with accurate colors
- **Output**: Flat vector-style PNGs with 9-color palette (100% exact colors, zero noise)

## Files in This Repository

### ✅ USE THESE
- **`START_HERE.bat`** ← **DOUBLE-CLICK THIS TO RUN**
- `restore_playmat_hsv.py` - Main HSV-based restoration script
- `README.md` - This file

### 📁 Reference Materials (Optional)
- `archive/docs/` - Technical documentation
- `archive/scripts/` - Legacy implementations for reference

## Troubleshooting

**Window closes immediately**: Make sure you have images in the `scans/` folder first

**Python not found**: Install Python 3.7+ from python.org and check "Add Python to PATH"

**Colors look wrong**: Use START_HERE.bat which runs the correct HSV version

---

**Ready to start?** Put images in `scans/` folder and double-click `START_HERE.bat`!
