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
    'black':         (0, 0, 0)         # Void/Scan Edges
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
    
    # ==== 1. WHITE ELEMENTS (stars, logos, text) ====
    # White has low saturation and high value
    # CRITICAL: Must be lenient to catch blue-tinted whites from lighting
    # Range: S < 80 (was 50), V > 180 (was 200) - catches RGB(205-250, 227-255, 245-255)
    white_mask = (s < 80) & (v > 180)
    img_processed[white_mask] = pure_white
    white_count = np.sum(white_mask)
    print(f"  White elements: {white_count:,} pixels → pure_white")
    
    # ==== 2. YELLOW ELEMENTS (silhouettes, text, with glare variations) ====
    # Yellow hue in HSV: 20-40 degrees (out of 180 in OpenCV)
    # Covers all yellow variations (silhouettes, text, glare)
    yellow_mask = (h >= 20) & (h <= 40) & (s > 100) & (v > 100)
    img_processed[yellow_mask] = bright_yellow
    yellow_count = np.sum(yellow_mask)
    print(f"  Yellow elements: {yellow_count:,} pixels → bright_yellow")
    
    # ==== 3. NEON GREEN (outlines around silhouettes) ====
    # Green hue: 40-80 degrees
    # CRITICAL: Don't be too restrictive - actual green outlines need to be preserved
    # Lower saturation threshold to catch all green variations
    green_mask = (h >= 40) & (h <= 80) & (s > 60) & (v > 80)
    img_processed[green_mask] = neon_green
    green_count = np.sum(green_mask)
    print(f"  Neon green outlines: {green_count:,} pixels → neon_green")
    
    # ==== 4. PINK/MAGENTA ELEMENTS (hot pink and dark purple) ====
    # Pink/Magenta hue: 140-170 degrees (wrapped around 180)
    # CRITICAL: Logo has 3 layers - white (already done), hot pink, dark purple
    pink_hue_mask = ((h >= 140) & (h <= 170)) & (s > 80)
    
    # Differentiate by value to preserve logo sandwich:
    # Hot pink (bright): V > 160 (lowered from 180)
    # Dark purple (outer layer): 100 < V <= 160
    hot_pink_mask = pink_hue_mask & (v > 160)
    dark_purple_mask = pink_hue_mask & (v > 80) & (v <= 160)
    
    img_processed[hot_pink_mask] = hot_pink
    img_processed[dark_purple_mask] = dark_purple
    
    pink_count = np.sum(hot_pink_mask)
    purple_count = np.sum(dark_purple_mask)
    print(f"  Hot pink: {pink_count:,} pixels → hot_pink")
    print(f"  Dark purple: {purple_count:,} pixels → dark_purple")
    
    # ==== 5. RED ELEMENTS (vibrant red) ====
    # Red hue: 0-10 or 170-180 degrees (wraps around)
    red_mask = ((h >= 0) & (h <= 10) | (h >= 170) & (h <= 180)) & (s > 150) & (v > 200)
    img_processed[red_mask] = vibrant_red
    red_count = np.sum(red_mask)
    print(f"  Vibrant red: {red_count:,} pixels → vibrant_red")
    
    # ==== 6. BLUE BACKGROUND (all variations: clean, glare, dirt) ====
    # Blue hue: 90-130 degrees
    # CRITICAL: Must NOT overwrite whites! Only process pixels not already white
    # Increase saturation threshold to avoid catching low-sat (white-ish) blues
    blue_mask = (h >= 90) & (h <= 130) & (s > 50) & (v > 80)
    # Don't overwrite whites
    blue_mask = blue_mask & (s >= 80)  # Exclude low-saturation (white) pixels
    img_processed[blue_mask] = sky_blue
    blue_count = np.sum(blue_mask)
    print(f"  Blue background (all): {blue_count:,} pixels → sky_blue")
    
    # ==== 7. BLACK/DEADSPACE ====
    # Very low value
    black_mask = (v < 30)
    img_processed[black_mask] = black
    black_count = np.sum(black_mask)
    print(f"  Black/deadspace: {black_count:,} pixels → black")
    
    return img_processed


def bilateral_smooth_edges(img, d=9, sigma_color=75, sigma_space=75):
    """Apply bilateral filter to smooth texture while preserving edges."""
    print("Applying bilateral filter for edge-preserving smoothing...")
    smoothed = cv2.bilateralFilter(img, d, sigma_color, sigma_space)
    return smoothed


def morphological_cleanup(img, kernel_size=3):
    """Apply morphological operations to clean up noise."""
    print(f"Applying morphological cleanup (kernel={kernel_size})...")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Closing to fill small holes
    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Opening to remove small noise
    cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
    
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
