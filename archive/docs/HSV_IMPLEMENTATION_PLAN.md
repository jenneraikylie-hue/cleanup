# HSV-Based Color Detection Implementation Plan

## Problem Analysis

Current BGR threshold-based approach has several issues:
1. **Channel confusion**: BGR vs RGB notation inconsistencies in comments and code
2. **Lighting sensitivity**: Direct BGR thresholds affected by blue-biased lighting
3. **Premature quantization**: Hard color snapping before noise removal
4. **Loss of detail**: Aggressive filtering removes fine features (stars, text, outlines)

## Solution: HSV Color Space

### Why HSV?
- **Hue (H)**: Color type, independent of lighting (0-180° in OpenCV)
- **Saturation (S)**: Color purity (0-255)
- **Value (V)**: Brightness (0-255)

Blue-biased lighting affects V and S, but H remains stable.

### HSV Ranges for Playmat Colors

#### 1. Sky Blue Background
- **Clean background**: H=100-115, S=40-80, V=180-255
- **Blue glare**: H=100-115, S=20-50, V=220-255 (desaturated, bright)
- **Blue dirt**: H=100-115, S=30-60, V=140-200 (darker)

#### 2. Yellow Silhouettes
- **Dense yellow**: H=25-35, S=200-255, V=195-255
- **Yellow text**: H=25-32, S=220-255, V=240-255 (brighter)
- **Yellow glare**: H=28-35, S=180-220, V=250-255 (slightly desaturated)

#### 3. Pink/Magenta
- **Hot pink main**: H=145-155, S=240-255, V=245-255
- **Pink bright outline**: H=145-155, S=220-255, V=235-255
- **Dark purple outline**: H=145-155, S=200-240, V=200-235 (darker value)

#### 4. Red
- **Vibrant red**: H=0-5 or H=175-180, S=240-255, V=245-255

#### 5. Neon Green (Edges Only)
- **Lime green outline**: H=35-50, S=180-255, V=180-215

#### 6. White Elements
- **Near-white**: S=0-30, V=205-255 (low saturation, high brightness)
- Must differentiate from blue glare using saturation

#### 7. Black/Deadspace
- **Black**: V=0-20 (any H, any S)

## Processing Pipeline

### Phase 1: Load and Prepare
```python
1. Load image
2. Upscale 3x (INTER_CUBIC)
3. Convert to HSV
4. Apply bilateral filter (preserve edges while reducing noise)
```

### Phase 2: Protection Masking
```python
1. Detect stars (HSV white mask + shape detection)
2. Detect text (HSV white mask + contrast + area)  
3. Detect logo regions (HSV pink/white/purple masks)
4. Combine into protection_mask
```

### Phase 3: Color Preprocessing (HSV-based)
```python
For each color category in HSV:
    1. Create HSV range mask
    2. Apply morphological opening (remove noise)
    3. Fill small holes (closing operation)
    4. Set pixels to target BGR palette color
```

**Order matters** (process from most specific to most general):
1. White elements (low saturation)
2. Yellow variations (by brightness)
3. Pink/purple (by brightness)
4. Red
5. Neon green (edge detection)
6. Blue background (clean, glare, dirt)
7. Black

### Phase 4: Edge Preservation
```python
1. Apply bilateral filter again (smooth regions, keep edges)
2. Detect edges with Canny
3. Protect edge pixels during solidification
```

### Phase 5: Color Region Solidification
```python
For each palette color (except white):
    1. Create mask for that color
    2. Apply median filter (kernel=5) within mask
    3. Replace with solidified color
```

### Phase 6: Final Quantization
```python
1. Snap all remaining pixels to nearest palette color
2. Downscale to original size (INTER_AREA for anti-aliasing)
3. One final palette snap (ensure 100% exact colors)
```

## Key Differences from Current Approach

| Current (BGR) | New (HSV) |
|--------------|-----------|
| Hard BGR thresholds | HSV ranges by hue/saturation/value |
| Early quantization | Late quantization after noise removal |
| Channel confusion | Clear color separation |
| Lighting-dependent | Lighting-independent hue |
| Aggressive median (kernel=11) | Gentler approach (kernel=5) + morphology |
| Green scattered | Green edge-detected only |

## Expected Improvements

1. **Accurate colors**: HSV hue isolates colors regardless of lighting
2. **Clean output**: Morphological operations remove speckles before quantization
3. **Preserved details**: White stars/text/logos not bleeding into blue
4. **Logo layers**: White/pink/dark purple properly differentiated by value
5. **Green outlines**: Edge detection ensures green only on boundaries
6. **No posterization**: Bilateral filter maintains smooth gradients before final snap

## Implementation Status

- [ ] Create HSV color range definitions
- [ ] Rewrite preprocess_special_colors() to use HSV
- [ ] Add morphological noise removal
- [ ] Implement edge-based green detection
- [ ] Test on sample images
- [ ] Validate all 9 palette colors present
- [ ] Verify 100% exact palette match
