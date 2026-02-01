# Vinyl Playmat Digital Restoration Tool

A robust Python script for digitally restoring high-resolution scans of vintage vinyl playmats. Removes wrinkles, glare, and texture while preserving logos, text, stars, and silhouettes with accurate colors.

## Features

- **Background Replacement**: Completely flattens the vinyl texture by replacing the sky blue background with a single solid color
- **Object Protection**: Automatically detects and preserves:
  - Stars (5-pointed white polygons)
  - Small text (numbers, instructions, copyright)
  - Logo layers (white/pink/purple sandwich structure)
- **Glare Removal**: Eliminates specular highlights and reflections
- **Wrinkle Removal**: Removes shadows and texture from the background
- **Color Quantization**: Maps all colors to a specific palette using Euclidean distance
- **Edge Preservation**: Uses bilateral filtering to maintain sharp outlines
- **Anti-Aliasing**: Smooths color boundaries to prevent jagged edges

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
    'deep_teal':     (10, 176, 149),   # Small Text/Shadows
    'black':         (0, 0, 0)         # Void/Scan Edges
}
```

## Requirements

- Python 3.7 or later
- OpenCV (cv2)
- NumPy

## Installation

### Automatic (Windows)

1. Double-click `run_cleanup.bat`
2. The script will automatically install dependencies if needed

### Manual

```bash
pip install opencv-python numpy
```

## Usage

### Quick Start (Windows)

1. Place your JPG images in the same folder as the scripts
2. Double-click `run_cleanup.bat`
3. Restored PNG files will be saved to the `restored` folder

### Command Line Usage

**Process all JPG images in current directory:**
```bash
python restore_playmat.py
```

**Process a specific image:**
```bash
python restore_playmat.py input.jpg output.png
```

**Process all images in a specific directory:**
```bash
python restore_playmat.py /path/to/images /path/to/output
```

## Processing Pipeline

### Phase 1: Pre-Processing & Detection
1. Load image and upscale 3x for better processing
2. Detect and protect stars (5-pointed white polygons)
3. Detect and protect text (high-contrast small details)
4. Detect and protect logos (pink/white/purple regions)

### Phase 2: Background Cleaning
1. Identify sky blue background region using LAB color space
2. Remove protected areas from background mask
3. Replace entire background with flat sky_blue color (233, 180, 130)

### Phase 3: Color Quantization & Restoration
1. Apply bilateral filter for edge-preserving smoothing
2. Snap all pixels to nearest palette color (vectorized operation)
3. Reinforce neon green outlines on silhouettes
4. Fill single-pixel holes (salt-and-pepper noise removal)

### Phase 4: Final Polish
1. Apply anti-aliasing to color boundaries
2. Downscale back to original resolution
3. Save as lossless PNG

## Special Handling

### The "Steps" Logo (Sandwich Structure)
The logo has three distinct layers that are preserved:
- **Center**: Pure White
- **Middle**: Hot Pink
- **Outer**: Dark Purple (kept separate from pink)

### Stars
- Detected by shape (5 points, low solidity)
- Differentiated from glare (which is irregular)
- Protected from background replacement

### Silhouettes
- Yellow fill with neon green outline
- Outline thickness preserved with bilateral filtering
- Interior solidly filled with no gradients

### Text
- Small high-contrast regions protected
- Not eroded by morphological operations
- Remains legible after processing

## Output

- Format: PNG (lossless compression)
- Resolution: Same as input
- Color depth: 8-bit per channel
- Result: Clean, flat-color "vector-style" image with no texture or artifacts

## Performance Tips

- Images are automatically upscaled 3x during processing for better accuracy
- Processing time depends on image resolution (typically 30-60 seconds per image)
- Large batches can be processed unattended

## Troubleshooting

**"Python is not installed"**
- Install Python from https://www.python.org/
- Make sure to check "Add Python to PATH" during installation

**"Failed to install dependencies"**
- Manually run: `pip install opencv-python numpy`
- You may need to use `pip3` instead of `pip` on some systems

**Output has artifacts**
- Ensure input images are high resolution (at least 1000x1000 pixels)
- Check that the input format is JPG

**Colors don't match palette**
- The script automatically snaps all colors to the nearest palette color
- If results are unexpected, check the lighting and exposure of your scans

## License

This tool is provided as-is for image restoration purposes.
