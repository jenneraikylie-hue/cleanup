# HSV-Based Color Detection Approach

## Problem with BGR Thresholds

The previous BGR threshold approach had issues with:
1. **Blue-biased lighting**: All colors shifted toward blue in scans
2. **Channel confusion**: Manual BGR conversions from RGB observations were error-prone
3. **Overlapping ranges**: White, blue background, and blue glare had overlapping BGR values
4. **Posterization**: Early quantization destroyed subtle gradients needed for proper detection

## HSV Color Space Advantages

HSV separates color information into three independent components:
- **Hue (H)**: The actual color (0-180° in OpenCV)
- **Saturation (S)**: Color purity (0-255, low=gray/white, high=vivid)
- **Value (V)**: Brightness (0-255, low=dark, high=bright)

### Why HSV Solves Our Problems

1. **Lighting invariance**: Hue remains constant under varying lighting intensity
2. **Natural color ranges**: Yellow is always hue 20-40° regardless of brightness
3. **Easy white detection**: White = low saturation + high value (any hue)
4. **Logo layer separation**: Same hue, different values = hot pink vs dark purple

## Color Detection Ranges (HSV)

### 1. White Elements (Stars, Logo Center, Text)
```python
Criteria: S < 50, V > 200
Logic: Low saturation (not colored) + high brightness
Handles: Blue-biased whites from lighting
```

### 2. Yellow Elements (Silhouettes, Text, Glare)
```python
Hue: 20-40° (yellow range)
Saturation: > 100 (actual color, not washed out)
Value: > 100 (visible, not too dark)
Handles: All yellow variations in one range
```

### 3. Neon Green (Outlines Only)
```python
Hue: 40-80° (green range)
Saturation: > 100
Additional: Edge detection (Canny) to restrict to outlines
Logic: Green only where edges exist
```

### 4. Pink/Magenta Elements (Logo)
```python
Hue: 140-170° (magenta range)
Saturation: > 100

Then differentiate by value:
- Hot Pink: V > 180 (bright)
- Dark Purple: 100 < V ≤ 180 (shadow/outline)

Preserves 3-layer logo: white/hot pink/dark purple
```

### 5. Red Elements (Outlines)
```python
Hue: 0-10° or 170-180° (red wraps around)
Saturation: > 150 (very saturated)
Value: > 200 (bright red)
```

### 6. Blue Background (All Variations)
```python
Hue: 90-130° (blue/cyan range)
Saturation: > 30 (any blue, even desaturated glare)
Handles: Clean background, glare, dirt all in one range
```

### 7. Black/Deadspace
```python
Value: < 30 (very dark)
```

## Pipeline Order (Delayed Quantization)

1. **HSV Preprocessing**: Detect colors, replace with palette
2. **Bilateral Filter**: Smooth textures while preserving edges
3. **Morphological Cleanup**: Remove small noise, fill holes
4. **Palette Snap**: Force all pixels to exact palette colors
5. **Solidify Regions**: Median filter within each color for flatness
6. **Downscale**: Return to original size
7. **Final Snap**: Enforce palette one more time

## Key Differences from BGR Approach

| Aspect | BGR Approach | HSV Approach |
|--------|--------------|--------------|
| White detection | B≥245, G≥227, R≥205, B>R | S<50, V>200 |
| Yellow detection | 3 separate ranges (silhouette/text/glare) | 1 range (H=20-40) |
| Pink layers | 3 separate ranges, value-blind | 1 hue, separated by value |
| Blue variations | 3 separate masks (clean/glare/dirt) | 1 range (H=90-130) |
| Green restriction | Complex edge detection + conversion | Edge detection during detection |
| Lighting tolerance | Poor (blue bias breaks thresholds) | Excellent (hue is lighting-invariant) |

## Expected Improvements

1. **No posterization**: Colors remain accurate under varying lighting
2. **Preserved layers**: Logo shows white/pink/purple distinctly
3. **Correct green**: Only on edges, not in fills
4. **Flat blue**: All background variations → single color
5. **Sharp edges**: Bilateral filter preserves boundaries
6. **No texture**: Solidification removes all material artifacts

## Testing Strategy

Compare output between BGR and HSV versions on same images:
- Check logo has 3 visible layers
- Verify green only on outlines
- Confirm no blue speckles in white
- Validate flat background (no texture/marbling)
- Ensure stars and text are clear
