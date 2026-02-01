#!/usr/bin/env python3
"""
Vinyl Playmat Digital Restoration Script - HSV-Based Implementation
Removes wrinkles, glare, and texture from scanned vinyl playmat images
while preserving logos, text, stars, and silhouettes with accurate colors.

This version uses HSV color space for robust color detection under varying lighting.
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path

# Master Color Palette (BGR Format for OpenCV)
PALETTE = {
    'sky_blue':      (233, 180, 130),  # Background (Flat Canvas)
    'hot_pink':      (205, 0, 253),    # Primary Logo/Footprints
    'bright_yellow': (1, 252, 253),    # Silhouettes/Ladder Rungs
    'pure_white':    (255, 255, 255),  # Stars/Logo Interior (PROTECTED)
    'neon_green':    (0, 213, 197),    # Silhouette Outlines
    'dark_purple':   (140, 0, 180),    # Outer Logo Border (3rd Layer)
    'vibrant_red':   (1, 13, 245),     # Ladder Accents/Underlines
    'deep_teal':     (10, 176, 149),   # Small Text/Shadows
    'black':         (0, 0, 0),        # Void/Scan Edges
    'outline_magenta': (149, 0, 219)   # Dark pink-purple outlines (logo, foot graphic, yellow block outlines)
}

# Convert palette to arrays for vectorized operations
PALETTE_ARRAY = np.array(list(PALETTE.values()), dtype=np.float32)
PALETTE_NAMES = list(PALETTE.keys())


def load_and_upscale(image_path, scale=3):
    """Load image and upscale for better processing."""
    print(f"Loading image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    original_size = (img.shape[1], img.shape[0])
    print(f"Original size: {original_size}")
    
    # Upscale 3x for better processing
    new_size = (img.shape[1] * scale, img.shape[0] * scale)
    img_large = cv2.resize(img, new_size, interpolation=cv2.INTER_CUBIC)
    print(f"Upscaled to: {new_size}")
    
    return img_large, original_size


def preprocess_with_hsv(img):
    """
    Pre-process image using HSV color space for robust color detection.
    This avoids BGR threshold confusion under blue-biased lighting.
    
    HSV Advantages:
    - Hue separates color from lighting intensity
    - Saturation differentiates colored objects from glare/white
    - Value differentiates light/dark versions of same hue
    """
    print("\n=== Pre-processing with HSV Color Detection ===")
    
    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Create output image
    img_processed = img.copy()
    b, g, r = cv2.split(img_processed)
    
    # Palette colors
    pure_white = np.array(PALETTE['pure_white'], dtype=np.uint8)
    sky_blue = np.array(PALETTE['sky_blue'], dtype=np.uint8)
    bright_yellow = np.array(PALETTE['bright_yellow'], dtype=np.uint8)
    hot_pink = np.array(PALETTE['hot_pink'], dtype=np.uint8)
    dark_purple = np.array(PALETTE['dark_purple'], dtype=np.uint8)
    vibrant_red = np.array(PALETTE['vibrant_red'], dtype=np.uint8)
    neon_green = np.array(PALETTE['neon_green'], dtype=np.uint8)
    black = np.array(PALETTE['black'], dtype=np.uint8)
    outline_magenta = np.array(PALETTE['outline_magenta'], dtype=np.uint8)
    
    # ==== 1. WHITE ELEMENTS FIRST (stars, logos, text, circular effects) ====
    # White has low saturation and high value
    # CRITICAL: Process white FIRST to preserve logo interiors, text, and circular effects
    # Relaxed thresholds (S < 60, V > 180) to catch all white elements including blue-tinted whites
    white_mask = (s < 60) & (v > 180)
    img_processed[white_mask] = pure_white
    white_count = np.sum(white_mask)
    print(f"  White elements: {white_count:,} pixels → pure_white")
    
    # ==== 2. BLUE BACKGROUND (after white, to avoid overwriting logo/text) ====
    # Blue hue: 85-135 degrees
    # CRITICAL: Process blue AFTER white, exclude already-white pixels
    # Only catch pixels with sufficient saturation that are clearly blue (not white)
    blue_mask = (h >= 85) & (h <= 135) & (s > 40) & (v > 50) & ~white_mask
    img_processed[blue_mask] = sky_blue
    blue_count = np.sum(blue_mask)
    print(f"  Blue background (all): {blue_count:,} pixels → sky_blue")
    
    # ==== 3. YELLOW ELEMENTS (silhouettes, text, with glare variations) ====
    # Yellow hue in HSV: 20-40 degrees (out of 180 in OpenCV)
    # Covers all yellow variations (silhouettes, text, glare)
    yellow_mask = (h >= 20) & (h <= 40) & (s > 100) & (v > 100)
    img_processed[yellow_mask] = bright_yellow
    yellow_count = np.sum(yellow_mask)
    print(f"  Yellow elements: {yellow_count:,} pixels → bright_yellow")
    
    # ==== 4. NEON GREEN (outlines around silhouettes) ====
    # Green hue: 35-85 degrees (expanded range)
    # CRITICAL: Expanded range and lowered thresholds to catch all green outline variations
    green_mask = (h >= 35) & (h <= 85) & (s > 40) & (v > 60)
    img_processed[green_mask] = neon_green
    green_count = np.sum(green_mask)
    print(f"  Neon green outlines: {green_count:,} pixels → neon_green")
    
    # ==== 5. PINK/MAGENTA ELEMENTS (hot pink, outline magenta, and dark purple) ====
    # Pink/Magenta hue: 140-175 degrees (expanded range for outline_magenta)
    # CRITICAL: Logo has layers - white, hot pink, outline_magenta, dark purple
    pink_hue_mask = ((h >= 140) & (h <= 175)) & (s > 70)
    
    # Differentiate by value to preserve all pink layers:
    # Hot pink (brightest): V > 200, very saturated - the main pink color
    # Outline magenta (medium): 120 < V <= 200 - rgb(219, 0, 149) darker outlines
    # Dark purple (darkest): 60 < V <= 120 - outer border
    hot_pink_mask = pink_hue_mask & (v > 200)
    outline_magenta_mask = pink_hue_mask & (v > 120) & (v <= 200)
    dark_purple_mask = pink_hue_mask & (v > 60) & (v <= 120)
    
    img_processed[hot_pink_mask] = hot_pink
    img_processed[outline_magenta_mask] = outline_magenta
    img_processed[dark_purple_mask] = dark_purple
    
    pink_count = np.sum(hot_pink_mask)
    magenta_count = np.sum(outline_magenta_mask)
    purple_count = np.sum(dark_purple_mask)
    print(f"  Hot pink: {pink_count:,} pixels → hot_pink")
    print(f"  Outline magenta: {magenta_count:,} pixels → outline_magenta")
    print(f"  Dark purple: {purple_count:,} pixels → dark_purple")
    
    # ==== 6. RED ELEMENTS (vibrant red) ====
    # Red hue: 0-10 or 170-180 degrees (wraps around)
    red_mask = (((h >= 0) & (h <= 10)) | ((h >= 170) & (h <= 180))) & (s > 150) & (v > 200)
    img_processed[red_mask] = vibrant_red
    red_count = np.sum(red_mask)
    print(f"  Vibrant red: {red_count:,} pixels → vibrant_red")
    
    # ==== 7. BLACK/DEADSPACE ====
    # Very low value
    black_mask = (v < 30)
    img_processed[black_mask] = black
    black_count = np.sum(black_mask)
    print(f"  Black/deadspace: {black_count:,} pixels → black")
    
    return img_processed


def bilateral_smooth_edges(img, d=15, sigma_color=100, sigma_space=100):
    """
    Apply bilateral filter to smooth texture while preserving edges.
    Increased parameters for stronger smoothing of pixelated outlines.
    """
    print("Applying bilateral filter for edge-preserving smoothing...")
    smoothed = cv2.bilateralFilter(img, d, sigma_color, sigma_space)
    return smoothed


def morphological_cleanup(img, kernel_size=5):
    """
    Apply morphological operations to clean up noise and smooth outlines.
    Removes small specs and smooths pixelated edges.
    """
    print(f"Applying morphological cleanup (kernel={kernel_size})...")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Opening first to remove small noise/specs (like white dots)
    opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Closing to fill small holes and smooth outlines
    cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    return cleaned


def snap_to_palette(img):
    """Snap every pixel to the nearest palette color using Euclidean distance."""
    print("Snapping to palette colors...")
    
    # Reshape image to (num_pixels, 3)
    pixels = img.reshape(-1, 3).astype(np.float32)
    
    # Compute distance to each palette color (vectorized)
    # Shape: (num_pixels, num_colors)
    distances = np.linalg.norm(pixels[:, np.newaxis, :] - PALETTE_ARRAY[np.newaxis, :, :], axis=2)
    
    # Find closest palette color for each pixel
    closest_indices = np.argmin(distances, axis=1)
    
    # Map to palette colors
    quantized_pixels = PALETTE_ARRAY[closest_indices].astype(np.uint8)
    
    # Reshape back to image
    quantized = quantized_pixels.reshape(img.shape)
    
    # Count pixels per color
    unique, counts = np.unique(closest_indices, return_counts=True)
    print("  Color distribution:")
    for idx, count in zip(unique, counts):
        pct = 100.0 * count / len(pixels)
        print(f"    {PALETTE_NAMES[idx]}: {count:,} pixels ({pct:.2f}%)")
    
    return quantized


def solidify_color_regions(img, kernel_size=5):
    """
    Apply median filter within each color region to remove gradients and texture.
    Smaller kernel (5 vs 11) to preserve detail better.
    """
    print(f"Solidifying color regions (median kernel={kernel_size})...")
    
    img_solid = img.copy()
    
    # Apply median blur once
    blurred = cv2.medianBlur(img, kernel_size)
    
    # For each palette color, find regions and apply solidification
    for color_name, color_bgr in PALETTE.items():
        # Skip white to prevent bleeding into blue
        if color_name == 'pure_white':
            continue
            
        color_bgr = np.array(color_bgr, dtype=np.uint8)
        
        # Find pixels of this color
        mask = np.all(img == color_bgr, axis=2)
        
        # Count region size
        num_pixels = np.sum(mask)
        
        # Only process if region is significant
        if num_pixels > 500:
            # Replace with median-blurred version
            img_solid[mask] = blurred[mask]
    
    return img_solid


def restore_image(image_path, output_dir):
    """Main restoration pipeline."""
    print(f"\n{'='*60}")
    print(f"Processing: {Path(image_path).name}")
    print(f"{'='*60}")
    
    # Phase 1: Load and upscale
    img_large, original_size = load_and_upscale(image_path)
    
    # Phase 2: HSV-based color preprocessing
    img_preprocessed = preprocess_with_hsv(img_large)
    
    # Phase 3a: Bilateral filtering for edge-preserving smoothing
    img_smooth = bilateral_smooth_edges(img_preprocessed)
    
    # Phase 3b: Morphological cleanup
    img_cleaned = morphological_cleanup(img_smooth)
    
    # Phase 3c: Snap to exact palette colors
    img_quantized = snap_to_palette(img_cleaned)
    
    # Phase 3d: Solidify color regions (remove any remaining texture)
    img_solid = solidify_color_regions(img_quantized)
    
    # Phase 4: Downscale back to original size
    print(f"Downscaling to original size: {original_size}")
    img_final = cv2.resize(img_solid, original_size, interpolation=cv2.INTER_AREA)
    
    # Phase 5: Final palette enforcement
    img_final = snap_to_palette(img_final)
    
    # Save output
    input_filename = Path(image_path).stem
    output_path = os.path.join(output_dir, f"{input_filename}_cleaned.png")
    cv2.imwrite(output_path, img_final)
    print(f"\nSaved to: {output_path}")
    
    return img_final


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python restore_playmat_hsv.py <image_or_directory>")
        print("Example: python restore_playmat_hsv.py scan.jpg")
        print("Example: python restore_playmat_hsv.py scans/")
        sys.exit(1)
    
    input_path = sys.argv[1]
    
    # Create output directory
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process single image or directory
    if os.path.isfile(input_path):
        restore_image(input_path, output_dir)
    elif os.path.isdir(input_path):
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(Path(input_path).glob(ext))
        
        print(f"Found {len(image_files)} images to process")
        
        for img_path in image_files:
            try:
                restore_image(str(img_path), output_dir)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
    else:
        print(f"Error: {input_path} is not a valid file or directory")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print("Processing complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
