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
    
    # ==== 3. NEON GREEN (outlines around silhouettes) - PROCESS BEFORE YELLOW ====
    # Based on provided color samples:
    # - Lime green outline: HSL 67-70° → OpenCV hue ~33-35
    # - Colors like #96be45, #b5cd00, #a9cb1b have hue around 33-35 in OpenCV scale
    # CRITICAL: Process green BEFORE yellow to preserve outline detail
    # Green range: 33-85° to catch lime-green outlines (HSL 67+), lowered to 33 for better coverage
    # Lowered saturation threshold to catch slightly desaturated green pixels
    green_mask = (h >= 33) & (h <= 85) & (s > 25) & (v > 45)
    
    # Apply morphological closing to fill gaps in green outlines
    # This creates consistent, continuous outlines without jaggedness
    green_mask_uint8 = green_mask.astype(np.uint8) * 255
    green_close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    green_mask_closed = cv2.morphologyEx(green_mask_uint8, cv2.MORPH_CLOSE, green_close_kernel, iterations=2)
    green_mask_final = green_mask_closed > 0
    
    img_processed[green_mask_final] = neon_green
    green_count = np.sum(green_mask_final)
    print(f"  Neon green outlines: {green_count:,} pixels → neon_green")
    
    # ==== 4. YELLOW ELEMENTS (silhouettes, blocks, ladder text, with glare variations) ====
    # Based on provided color samples:
    # - Block yellow (infill): HSL 59-60° → OpenCV hue ~29-30 (#FBFB00, #FAF900, #F8F800)
    #   Very high saturation (100%), high value (49%)
    # - Golden orange: HSL 58-59° → OpenCV hue ~29-30 (#FEF900, #FFFA00)
    # - Yellow silhouette inside: HSL 62-63° → OpenCV hue ~31 (#DFE801, #DCE803)
    # Yellow hue in HSV: 20-33 degrees to capture all yellow variations including block yellow
    # CRITICAL: Process AFTER green to preserve green outlines around yellow silhouettes
    # Exclude pixels already marked as green
    yellow_mask = (h >= 20) & (h <= 33) & (s > 80) & (v > 100) & ~green_mask_final
    img_processed[yellow_mask] = bright_yellow
    yellow_count = np.sum(yellow_mask)
    print(f"  Yellow elements: {yellow_count:,} pixels → bright_yellow")
    
    # ==== 5. PINK/MAGENTA ELEMENTS (hot pink, outline magenta, and dark purple) ====
    # Based on provided color samples:
    # - Hot pink/Magenta: HSL 311-315° → OpenCV hue ~155-158 (#FF00C9, #FD00CC, #F600B8)
    #   Very high saturation, V > 240 (RGB values near 255)
    # - Outline magenta (dark pinkish-purple outline): HSL 308-317° → OpenCV hue ~154-159
    #   Colors like #EC00AA, #E801B5, #EF00B3 with V ~220-240
    #   This is the thin line that sits outside the hot pink
    # - Dark purple (darkest): V < 180 - outer border
    # Pink/Magenta hue: 140-175 degrees (covers hot pink through outline_magenta)
    # CRITICAL: Logo has layers - white, hot pink, outline_magenta, dark purple
    pink_hue_mask = ((h >= 140) & (h <= 175)) & (s > 70)
    
    # Differentiate by value to preserve all pink layers:
    # Hot pink (brightest): V > 240, very saturated - the main pink color (#FF00C9, etc.)
    # Outline magenta (medium-high): 180 < V <= 240 - darker pink-purple outlines (#EC00AA, rgb(219,0,149))
    # Dark purple (darkest): 50 < V <= 180 - outer border
    hot_pink_mask = pink_hue_mask & (v > 240)
    outline_magenta_mask = pink_hue_mask & (v > 180) & (v <= 240)
    dark_purple_mask = pink_hue_mask & (v > 50) & (v <= 180)
    
    img_processed[hot_pink_mask] = hot_pink
    img_processed[outline_magenta_mask] = outline_magenta
    img_processed[dark_purple_mask] = dark_purple
    
    pink_count = np.sum(hot_pink_mask)
    magenta_count = np.sum(outline_magenta_mask)
    purple_count = np.sum(dark_purple_mask)
    print(f"  Hot pink: {pink_count:,} pixels → hot_pink")
    print(f"  Outline magenta: {magenta_count:,} pixels → outline_magenta")
    print(f"  Dark purple: {purple_count:,} pixels → dark_purple")
    
    # ==== 6. RED ELEMENTS (vibrant red outlines) ====
    # Based on provided color samples:
    # - Red outline colors: HSL 345-360° and 0-3° → OpenCV hue ~173-180 and 0-2
    # - Colors like #F90208, #FA0113, #FA013F, #FD0108, #FB0020 (outline reds)
    # - Also #FC0100, #FE0001, #FA1D1D (solid reds)
    # - All have very high saturation (98-100%) and medium-high value (48-50%)
    # Red hue: 0-12 or 168-180 degrees (wraps around, extended to catch #FA013F at HSL 345°)
    # Lowered saturation threshold to catch slightly desaturated reds
    red_mask = (((h >= 0) & (h <= 12)) | ((h >= 168) & (h <= 180))) & (s > 100) & (v > 140)
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


def detect_text_regions(img):
    """
    Detect text-like regions for protection.
    Text on playmats is:
    - WHITE for instruction text
    - LIME GREEN for heading text
    
    Uses color-based detection combined with edge analysis.
    Returns a mask of text regions that should be protected.
    """
    print("Detecting text regions for protection (white instructions, lime green headings)...")
    
    # Convert to HSV for color-based text detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # === WHITE TEXT DETECTION (instruction text) ===
    # White has low saturation and high value
    white_text_mask = (s < 60) & (v > 180)
    
    # === LIME GREEN TEXT DETECTION (heading text) ===
    # Lime green hue: 33-85 in OpenCV's 0-179 scale (matches neon_green detection)
    # Based on color samples: HSL 67-70° → OpenCV ~33-35
    # Medium-high saturation and value
    lime_green_mask = (h >= 33) & (h <= 85) & (s > 30) & (v > 100)
    
    # Combine color masks for text colors
    text_color_mask = (white_text_mask | lime_green_mask).astype(np.uint8) * 255
    
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Canny edge detection to find text edges
    edges = cv2.Canny(gray, 50, 150)
    
    # Dilate edges to connect text strokes into regions
    text_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    dilated = cv2.dilate(edges, text_kernel, iterations=2)
    
    # Find contours of potential text regions
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create text protection mask based on contours
    contour_mask = np.zeros(gray.shape, dtype=np.uint8)
    
    # Pre-calculate max text area threshold (10% of image area)
    max_text_area = img.shape[0] * img.shape[1] * 0.1
    
    for contour in contours:
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Skip contours with zero height (horizontal lines)
        if h == 0:
            continue
            
        area = cv2.contourArea(contour)
        
        # Text characteristics: reasonable aspect ratio, not too small, not too large
        aspect_ratio = w / h
        
        # Text regions typically have aspect ratio > 1 (wider than tall) or 
        # are small enough to be individual characters
        is_text_like = (
            (0.1 < aspect_ratio < 20) and  # Reasonable aspect ratio
            (area > 100) and  # Not too small (noise)
            (area < max_text_area) and  # Not too large (background)
            (w > 10 or h > 10)  # Minimum dimension
        )
        
        if is_text_like:
            # Add padding around detected text region
            padding = 5
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(img.shape[1], x + w + padding)
            y2 = min(img.shape[0], y + h + padding)
            contour_mask[y1:y2, x1:x2] = 255
    
    # Also protect high-contrast thin elements (likely text strokes)
    # Use morphological gradient to find edges
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, morph_kernel)
    
    # Threshold gradient to find high-contrast regions
    _, high_contrast = cv2.threshold(gradient, 30, 255, cv2.THRESH_BINARY)
    
    # Dilate to create protection buffer around edges
    edge_buffer = cv2.dilate(high_contrast, morph_kernel, iterations=2)
    
    # Combine all detection methods:
    # 1. Text color regions (white and lime green)
    # 2. Contour-based text detection
    # 3. High-contrast edge regions
    combined_mask = cv2.bitwise_or(text_color_mask, contour_mask)
    combined_mask = cv2.bitwise_or(combined_mask, edge_buffer)
    
    # Dilate the combined mask slightly to create a protection buffer
    protection_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    protected_mask = cv2.dilate(combined_mask, protection_kernel, iterations=1)
    
    text_pixel_count = np.sum(protected_mask > 0)
    total_pixels = img.shape[0] * img.shape[1]
    print(f"  Text protection: {text_pixel_count:,} pixels ({100.0 * text_pixel_count / total_pixels:.2f}%)")
    
    return protected_mask


def morphological_cleanup(img, kernel_size=5, text_mask=None):
    """
    Apply morphological operations to clean up noise and smooth outlines.
    Removes small specs and smooths pixelated edges.
    If text_mask is provided, applies gentler processing to text regions.
    """
    print(f"Applying morphological cleanup (kernel={kernel_size})...")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Opening first to remove small noise/specs (like white dots)
    opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Closing to fill small holes and smooth outlines
    cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # If text mask provided, blend original with cleaned to preserve text
    if text_mask is not None:
        print("  Applying text protection - preserving original in text regions...")
        # Use smaller kernel for text regions (gentler cleanup)
        text_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        text_cleaned = cv2.morphologyEx(img, cv2.MORPH_CLOSE, text_kernel, iterations=1)
        
        # Create 3-channel mask for blending
        text_mask_3ch = cv2.cvtColor(text_mask, cv2.COLOR_GRAY2BGR)
        text_mask_float = text_mask_3ch.astype(np.float32) / 255.0
        
        # Blend: use gentler cleanup for text regions, aggressive cleanup elsewhere
        cleaned = (text_mask_float * text_cleaned + (1 - text_mask_float) * cleaned).astype(np.uint8)
    
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


def edge_preserving_smooth(img, sigma_color=75, sigma_space=75):
    """
    Apply edge-preserving smoothing to reduce jaggedness on outlines
    without filling in or expanding color regions.
    Uses bilateral filter which preserves edges while smoothing within regions.
    """
    print("Applying edge-preserving smoothing for clean outlines...")
    
    # Bilateral filter smooths within regions while preserving edges
    # This is less aggressive than contour-based smoothing and won't fill areas
    smoothed = cv2.bilateralFilter(img, d=9, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    
    return smoothed


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
    
    # Phase 2: Detect text regions for protection BEFORE any processing
    text_mask = detect_text_regions(img_large)
    
    # Phase 3: HSV-based color preprocessing
    img_preprocessed = preprocess_with_hsv(img_large)
    
    # Phase 4a: Bilateral filtering for edge-preserving smoothing
    img_smooth = bilateral_smooth_edges(img_preprocessed)
    
    # Phase 4b: Morphological cleanup with text protection
    img_cleaned = morphological_cleanup(img_smooth, text_mask=text_mask)
    
    # Phase 4c: Snap to exact palette colors
    img_quantized = snap_to_palette(img_cleaned)
    
    # Phase 4d: Edge-preserving smooth to reduce jaggedness without filling areas
    img_smooth_outlines = edge_preserving_smooth(img_quantized)
    
    # Phase 4e: Solidify color regions (remove any remaining texture)
    img_solid = solidify_color_regions(img_smooth_outlines)
    
    # Phase 5: Downscale back to original size
    print(f"Downscaling to original size: {original_size}")
    img_final = cv2.resize(img_solid, original_size, interpolation=cv2.INTER_AREA)
    
    # Phase 6: Final palette enforcement
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
