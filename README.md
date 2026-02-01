# Vinyl Playmat Digital Restoration Tool

A Python script for digitally restoring high-resolution scans of vintage vinyl playmats. Removes wrinkles, glare, and texture while preserving logos, text, stars, and silhouettes with accurate colors.

## Two Implementations Available

### 🌟 HSV-Based Detection (Recommended) - `restore_playmat_hsv.py`

Uses HSV color space for robust, lighting-invariant color detection.

**Key Advantages:**
- ✅ No posterization or harsh color shifts
- ✅ Correct under varying lighting (handles blue-biased scans)
- ✅ Natural logo layer separation (white/pink/purple) by brightness
- ✅ Simpler ranges (one per color family)
- ✅ Better edge preservation

**Usage:**
```bash
python restore_playmat_hsv.py scan.jpg          # Single image
python restore_playmat_hsv.py scans/            # Whole directory
```

📖 **Technical Details:** See [HSV_APPROACH.md](HSV_APPROACH.md)

### 🔧 BGR Threshold Detection (Legacy) - `restore_playmat.py`

Uses direct BGR channel thresholds based on measured ranges from actual scans.

**Characteristics:**
- 13 source color categories → 9 palette colors
- Separate ranges for variations (glare, dirt, clean background)
- May struggle with lighting variation
- Backup implementation for reference

**Usage:**
```bash
python restore_playmat.py scan.jpg              # Single image
python restore_playmat.py scans/                # Whole directory
```

## Features (Both Versions)

- **Flat Color Output**: Perfect vector-style appearance with zero texture
- **Object Protection**: Preserves stars, text, and logos via shape detection
- **Logo Preservation**: Maintains 3-layer structure (white/pink/purple)
- **Edge-Preserving Smoothing**: Bilateral filtering keeps outlines sharp
- **Color Quantization**: 9-color palette with exact color matching
- **Batch Processing**: Process directories of images automatically
- **3x Upscale Workflow**: Internal processing at 3x resolution for quality

## Master Color Palette (BGR Format)

```python
PALETTE = {
    'sky_blue':      (233, 180, 130),  # Background
    'hot_pink':      (205, 0, 253),    # Primary Logo/Footprints
    'bright_yellow': (1, 252, 253),    # Silhouettes/Ladder Rungs
    'pure_white':    (255, 255, 255),  # Stars/Logo Interior
    'neon_green':    (0, 213, 197),    # Silhouette Outlines
    'dark_purple':   (140, 0, 180),    # Outer Logo Border
    'vibrant_red':   (1, 13, 245),     # Ladder Accents
    'deep_teal':     (10, 176, 149),   # Small Text
    'black':         (0, 0, 0)         # Deadspace
}
```

## Requirements

- Python 3.7+
- OpenCV
- NumPy

**Install:**
```bash
pip install opencv-python numpy
```

## Windows Quick Launch

```bash
run_cleanup.bat  # Auto-installs dependencies, runs HSV version
```

## Processing Pipeline (HSV Version)

### Phase 1: Load & Upscale
- Load image and upscale 3x for better processing

### Phase 2: HSV Color Detection
- Convert to HSV color space
- Detect colors by hue (yellow=20-40°, green=40-80°, pink=140-170°, etc.)
- Use saturation to separate white from blue background
- Use value to differentiate hot pink (bright) from dark purple (shadow)
- Edge detection restricts green to outlines only

### Phase 3: Cleaning & Quantization
- **3a**: Bilateral filter (edge-preserving smoothing)
- **3b**: Morphological cleanup (noise removal)
- **3c**: Snap to exact palette colors
- **3d**: Solidify regions with median filter (removes texture)

### Phase 4: Downscale & Finalize
- Downscale to original resolution
- Final palette enforcement for 100% color accuracy

## Output

- **Format**: PNG (lossless)
- **Quality**: 100% exact palette colors, zero texture/noise
- **Appearance**: Clean vector-style flat colors
- **Location**: `output/` directory with `_cleaned.png` suffix

## Documentation

- [HSV_APPROACH.md](HSV_APPROACH.md) - Technical explanation of HSV method
- [HSV_IMPLEMENTATION_PLAN.md](HSV_IMPLEMENTATION_PLAN.md) - Implementation planning
- [AI_REVIEW_IMPLEMENTATION.md](AI_REVIEW_IMPLEMENTATION.md) - Color range analysis
- [FIXES_SUMMARY.md](FIXES_SUMMARY.md) - Issue resolution history
- [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md) - Usage examples
- [SECURITY_SUMMARY.md](SECURITY_SUMMARY.md) - Security analysis

## Troubleshooting

**Posterized or wrong colors:**
- Use `restore_playmat_hsv.py` (recommended)
- Check that input images are well-lit scans

**Green appearing in block fills:**
- HSV version restricts green to edges only
- BGR version may need adjustment

**White elements turning blue:**
- HSV version handles this better (saturation-based detection)
- Ensures S<50 for white detection

**Logo layers not visible:**
- HSV version uses value-based separation
- Hot pink (V>180) vs dark purple (V≤180)

## Performance

- Typical: 30-60 seconds per image (depends on resolution)
- Batch processing supported for unattended operation
- 3x upscaling means memory usage scales with image size

## License

This tool is provided as-is for image restoration purposes.
