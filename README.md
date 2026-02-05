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
By default, images are processed **one at a time** for memory safety with large images. Enable parallel processing if you have smaller images and abundant RAM:
```bash
# Default: Sequential processing (1 image at a time - safest for large images)
python restore_playmat_hsv.py scans/

# Enable parallel processing (use cautiously - can cause crashes with large images)
python restore_playmat_hsv.py scans/ --workers 4
```

**⚠️ Important**: Each worker loads a full high-resolution image into memory. Since we're dealing with large images, the default is **1 worker** (sequential) to prevent memory exhaustion and system crashes. Only increase `--workers` if you have abundant RAM (32GB+) and smaller images.

### GPU Acceleration (CUDA)
If you have an NVIDIA GPU with CUDA support, enable GPU acceleration for faster processing:
```bash
# Enable GPU acceleration
python restore_playmat_hsv.py scans/ --use-gpu

# Combine with parallel processing
python restore_playmat_hsv.py scans/ --workers 8 --use-gpu
```

**Setting up GPU acceleration:**

The easiest way to enable GPU acceleration is using **CuPy** (a CUDA-accelerated NumPy library):

1. **Check your CUDA version**: Run `nvidia-smi` and look for "CUDA Version" (e.g., 12.2)
2. **Install CuPy** matching your CUDA version:
   ```bash
   # For CUDA 12.x (most modern systems)
   pip install cupy-cuda12x
   
   # For CUDA 11.x
   pip install cupy-cuda11x
   ```
3. **Run with GPU**: `python restore_playmat_hsv.py scans/ --use-gpu`

**Alternative: OpenCV with CUDA** (more complex, better performance for image operations):
- Build OpenCV from source with CUDA enabled
- See: https://docs.opencv.org/4.x/d6/d15/tutorial_building_tegra_cuda.html

The script automatically detects available GPU backends and will use the best one available. When you run with `--use-gpu`, it will display which backend is being used and provide diagnostic information if GPU is not available.

### Performance Options
| Option | Description |
|--------|-------------|
| `--workers N` | Number of parallel workers (default: 1 for memory safety with large images) |
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

**Out of memory**: The script uses tile-based processing to handle large images without memory exhaustion. Large images are automatically processed in smaller chunks (tiles) to keep memory usage under 1 GB per operation.

---

**Ready to start?** Put images in `scans/` folder and double-click `START_HERE.bat`!
