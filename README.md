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

## Performance Optimizations for High-Powered Computers

The script is optimized for high-powered computers with multiple CPU cores and optional GPU acceleration:

### Parallel Processing
Process multiple images simultaneously using multiple CPU cores:
```bash
# Auto-detect optimal worker count (default: 4 workers max for memory safety)
python restore_playmat_hsv.py scans/

# Specify number of parallel workers (use cautiously - high values can cause crashes)
python restore_playmat_hsv.py scans/ --workers 8
```

**⚠️ Important**: Each worker loads a full high-resolution image into memory. The default is capped at 4 workers to prevent memory exhaustion and system crashes. Only increase `--workers` if you have abundant RAM (32GB+) and are processing smaller images.

### GPU Acceleration (CUDA)
If you have an NVIDIA GPU with CUDA support, enable GPU acceleration for faster processing:
```bash
# Enable GPU acceleration
python restore_playmat_hsv.py scans/ --use-gpu

# Combine with parallel processing
python restore_playmat_hsv.py scans/ --workers 8 --use-gpu
```

### Performance Options
| Option | Description |
|--------|-------------|
| `--workers N` | Number of parallel workers (default: 4 max, auto-detect based on CPU cores) |
| `--use-gpu` | Enable CUDA/GPU acceleration if available |
| `--sequential` | Force sequential processing (disable parallelism) |

### System Requirements for Best Performance
- **CPU**: Multi-core processor (8+ cores recommended for batch processing)
- **RAM**: 16GB+ recommended for large high-resolution scans
- **GPU** (optional): NVIDIA GPU with CUDA support for accelerated processing
- **Storage**: SSD recommended for faster I/O with large image files

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

**Out of memory**: Large images with 3x upscaling require significant RAM. The script defaults to max 4 parallel workers to prevent crashes. Process fewer images at once or use `--workers 1` for very large images or systems with limited RAM.

---

**Ready to start?** Put images in `scans/` folder and double-click `START_HERE.bat`!
