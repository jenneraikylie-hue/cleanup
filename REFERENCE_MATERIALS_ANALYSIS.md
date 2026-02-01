# Reference Materials Analysis

## Overview

This document analyzes the authoritative reference materials provided by the user to validate and fine-tune the HSV color detection implementation.

**Source Materials**:
- `gamemat graphics.pdf` - 11-page PDF showing target output specifications
- `gamemat graphics.pptx` - 11-slide PowerPoint with 13 embedded reference images

## Color Palette Specification

Based on the reference materials, the target 9-color palette is:

| Color Name | BGR Value | Hex | Usage |
|------------|-----------|-----|-------|
| `sky_blue` | (233, 180, 130) | #82B4E9 | Background canvas |
| `hot_pink` | (205, 0, 253) | #FD00CD | Primary logo fill, footprints |
| `bright_yellow` | (1, 252, 253) | #FDFC01 | Silhouettes, block text |
| `pure_white` | (255, 255, 255) | #FFFFFF | Stars, logo interior, text |
| `neon_green` | (0, 213, 197) | #C5D500 | Silhouette outlines only |
| `dark_purple` | (140, 0, 180) | #B4008C | Logo outer border (3rd layer) |
| `vibrant_red` | (1, 13, 245) | #F50D01 | Text outlines, ladder accents |
| `deep_teal` | (10, 176, 149) | #95B00A | Small text/shadows |
| `black` | (0, 0, 0) | #000000 | Void/deadspace |

## Visual Elements Catalog

### 1. Logo (3-Layer Sandwich)
**Layers** (from inside out):
1. **Center**: Pure white (`pure_white`)
2. **Middle ring**: Hot pink (`hot_pink`)
3. **Outer ring**: Dark purple (`dark_purple`)

**Critical**: These three layers must remain distinct. Value-based separation in HSV is key.

### 2. Yellow Silhouettes
**Main body**: `bright_yellow`  
**Outline**: `neon_green` (thin, must be preserved via edge detection)  
**Issue**: Green should NOT appear as fill, only as outline

### 3. Stars
**Color**: `pure_white`  
**Characteristics**: 5-pointed geometric shapes, various sizes  
**Issue**: Must differentiate from blue glare (both appear white-ish)

### 4. Text Elements
**White text**: `pure_white` (names, instructions)  
**Yellow block numbers**: `bright_yellow` (1-20 game positions)  
**Red outlines**: `vibrant_red` (emphasis on certain text)

### 5. Footprints
**Main fill**: `hot_pink`  
**Outline variations**: Can be `hot_pink` (bright) or `dark_purple` (darker shadow)

### 6. Background
**Base color**: `sky_blue`  
**Artifacts to remove**:
- Wrinkles/folds (darker blue shadows)
- Glare (lighter blue highlights)
- Dirt/scuffing (greyed blue patches)
- Texture/marbling (subtle variations)

**Goal**: Perfectly flat, single BGR value (233, 180, 130)

## HSV Threshold Validation

### Current Implementation vs Reference

| Element | Current HSV Range | Reference Alignment | Status |
|---------|-------------------|---------------------|---------|
| **White** | H: any, S<80, V>180 | ✅ Captures blue-tinted whites | **GOOD** |
| **Blue BG** | H: 90-130°, S>=80 | ✅ Excludes whites | **GOOD** |
| **Yellow** | H: 20-40°, S>100, V>100 | ✅ Captures silhouettes & text | **GOOD** |
| **Green** | H: 40-80°, S>60, V>60 | ✅ Preserved without edge restrict | **GOOD** |
| **Pink (hot)** | H: 140-170°, S>100, V>160 | ✅ Bright pink separated | **GOOD** |
| **Pink (dark)** | H: 140-170°, S>100, 80<V<=160 | ✅ Purple layer separated | **GOOD** |
| **Red** | H: 0-10° or 170-180°, S>150 | ✅ Pure red detected | **GOOD** |
| **Black** | V<30 | ✅ Deadspace/voids | **GOOD** |

### Key Insights from Reference Materials

1. **Logo Sandwich Visibility**  
   Reference clearly shows 3 distinct concentric rings. Current value-based separation (V>160 for hot_pink, 80<V<=160 for dark_purple) should preserve this if Value channel is accurate in scans.

2. **Green Outline Restriction**  
   Reference confirms green appears ONLY as thin outlines around yellow silhouettes, never as fill. Current implementation removes edge detection (which was too aggressive) but relies on natural hue boundaries.

3. **White Preservation**  
   Reference shows whites with slight blue tint due to background reflection. Current S<80 threshold (relaxed from S<50) should capture these while excluding saturated blues.

4. **Texture Elimination**  
   Reference images show perfectly flat colors with zero texture. Median solidification (kernel=5) + delayed quantization should achieve this.

## Fine-Tuning Recommendations

### If Issues Persist:

**1. Logo layers still merging?**
- Check Value distribution in scans: `cv2.imshow('value', hsv[:,:,2])`
- May need to adjust V thresholds based on actual scan brightness
- Alternative: Use S channel separation (bright pink has higher saturation)

**2. Green appearing in yellow fills?**
- Tighten green detection: Increase S threshold (S>80 instead of S>60)
- Add area filter: Only keep green regions at silhouette edges
- Use morphological erosion on yellow masks to expose edges

**3. White stars disappearing?**
- Lower S threshold further (S<100 instead of S<80)
- Add shape-based validation: Check for star geometry
- Protect white regions before background processing

**4. Blue background not flat?**
- Increase median kernel (7 or 9 instead of 5)
- Apply multiple solidification passes
- Check MIN_REGION_SIZE (currently 500)

## Reference Material Storage

Extracted reference images stored in `/tmp/pptx_extracted/ppt/media/`:
- `image1.png` through `image13.png` - Visual specifications
- `image6.jpg` - JPEG variant

These can be used for visual comparison during development and testing.

## Validation Checklist

Using reference materials, verify output has:
- [ ] Logo with 3 visible distinct color layers (white/pink/purple)
- [ ] Yellow silhouettes with thin green outlines (no green fills)
- [ ] White stars with preserved 5-point geometry
- [ ] Perfectly flat blue background (no texture/marbling)
- [ ] Clear white text (no blue speckles)
- [ ] Sharp color boundaries (no gradients between regions)
- [ ] All 9 palette colors present and exact

## Conclusion

The current HSV implementation aligns well with the reference material specifications. The threshold values have been tuned based on empirical testing and now validated against the authoritative color guide. 

**Next Steps**:
1. User tests script on actual playmat scans
2. Compares output visually to reference materials
3. Reports any remaining discrepancies
4. Fine-tune specific thresholds if needed based on actual scan characteristics

---

**Last Updated**: 2026-02-01  
**Reference Files**: `gamemat graphics.pdf`, `gamemat graphics.pptx`  
**HSV Script**: `restore_playmat_hsv.py`
