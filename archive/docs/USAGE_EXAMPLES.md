# Usage Examples

This document provides practical examples of using the Vinyl Playmat Digital Restoration Tool.

## Quick Start Examples

### Example 1: Process a Single Image (Windows)

Simply drag and drop your JPG file onto `run_cleanup.bat`, or:

1. Place your image in the same folder as the scripts
2. Double-click `run_cleanup.bat`
3. Find the restored PNG in the `restored` folder

### Example 2: Process a Single Image (Command Line)

```bash
python restore_playmat.py my_playmat.jpg my_playmat_clean.png
```

### Example 3: Process All Images in Current Directory

```bash
python restore_playmat.py
```

This will:
- Find all `.jpg` files in the current directory
- Process each one
- Save outputs to `restored/` folder with `_restored.png` suffix

### Example 4: Process Images from Another Directory

```bash
python restore_playmat.py /path/to/scans /path/to/output
```

### Example 5: Batch Process with Custom Output Location

```bash
python restore_playmat.py . cleaned_images
```

This saves all restored images to the `cleaned_images` folder.

## Expected Results

### Before Processing:
- Wrinkled vinyl with shadows
- Glare/specular highlights
- Mixed blue tones in background
- Grainy texture
- Inconsistent colors

### After Processing:
- Flat, solid sky blue background (233, 180, 130 BGR)
- No glare or reflections
- Clean, sharp edges
- Colors quantized to exact palette values
- Vector-style appearance
- Lossless PNG format

## Typical Processing Time

- Small image (1000x1000): ~10-20 seconds
- Medium image (2000x2000): ~30-45 seconds  
- Large image (3000x3000): ~60-90 seconds

Processing includes:
1. 3x upscaling
2. Object detection and masking
3. Background replacement
4. Color quantization
5. Edge enhancement
6. Downscaling to original size

## File Naming

Input: `my_playmat.jpg`
Output: `my_playmat_restored.png`

The script automatically:
- Preserves the original filename
- Adds `_restored` suffix
- Converts to `.png` format

## Tips for Best Results

1. **High Resolution Scans**: Use at least 1500x1500 pixels for best results
2. **Good Lighting**: Even lighting reduces extreme shadows (though the script handles most issues)
3. **Flat Surface**: Try to minimize major folds during scanning
4. **Clean Lens**: Remove dust from scanner glass
5. **Multiple Passes**: Process test images first to verify settings

## Troubleshooting Common Issues

### Issue: Colors Don't Match Original

**Solution**: The script enforces the exact color palette. This is intentional to create a clean digital reproduction. The palette matches the original design intent, not the faded/dirty scan.

### Issue: Small Text Disappeared

**Solution**: Ensure input resolution is high enough (1500+ pixels). The script protects text, but extremely low resolution may not have enough detail.

### Issue: Processing Takes Too Long

**Solution**: Normal for large images. The 3x upscaling means a 3000x3000 image becomes 9000x9000 during processing. This is necessary for accuracy.

### Issue: Some Stars Are Missing

**Solution**: The script differentiates stars (geometric 5-point shapes) from glare (irregular). If a star is severely damaged in the scan, it may not be detected. This is rare.

## Advanced Usage

### Process Only Specific File Types

```bash
# Process only files matching a pattern
for file in scan_*.jpg; do
    python restore_playmat.py "$file"
done
```

### Check Processing Progress

The script outputs detailed progress information:
- Phase 1: Protection mask creation
- Phase 2: Background cleaning  
- Phase 3: Color quantization and enhancement
- Phase 4: Final polishing

Each phase shows pixel counts and completion status.

## Integration with Other Tools

The script outputs standard PNG files that work with:
- Image editors (Photoshop, GIMP, etc.)
- Vector conversion tools (for further vectorization)
- Printing services (high-quality reproduction)
- Web publishing (lossless compression)

## Quality Verification

After processing, check:

1. **Background**: Should be uniform sky blue with no texture
2. **Logo**: White center, pink middle, purple outer border
3. **Silhouettes**: Yellow fill with green outline
4. **Stars**: Clean 5-pointed white shapes
5. **Text**: Sharp and legible
6. **Edges**: Smooth but not blurry

## Performance Notes

The script uses:
- Vectorized NumPy operations (fast)
- Bilateral filtering (edge-preserving)
- Morphological operations (efficient)
- Multi-scale processing (3x upscale/downscale)

Memory usage scales with image size. For very large images (>5000x5000), ensure at least 4GB RAM is available.
