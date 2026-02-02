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
    # UPDATED: Lower saturation threshold (20) and Value (40) to catch glared green
    green_mask = (h >= 30) & (h <= 90) & (s > 20) & (v > 40)
    
    # ==== BLUE-FACING OUTLINE ONLY (Option A) ====
    # Green should only exist where it borders blue (background), not where it borders yellow
    # This makes green behave as a "halo/accent" rather than a structural border
    # Detect green pixels that touch yellow and erode only those
    # Keep green pixels that touch blue
    
    # Create masks for neighbor detection
    green_mask_uint8 = green_mask.astype(np.uint8) * 255
    
    # Create a "yellow-ish" hue mask (potential yellow interior pixels)
    yellow_hue_mask = (h >= 20) & (h <= 33) & (s > 60) & (v > 80)
    yellow_hue_mask_uint8 = yellow_hue_mask.astype(np.uint8) * 255
    
    # Dilate yellow to find green pixels that are adjacent to yellow
    yellow_neighbor_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    yellow_dilated = cv2.dilate(yellow_hue_mask_uint8, yellow_neighbor_kernel, iterations=1)
    
    # Green pixels that touch yellow (yellow-facing green)
    yellow_facing_green = green_mask_uint8 & yellow_dilated
    
    # Dilate blue to find green pixels that are adjacent to blue (background)
    blue_neighbor_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    blue_dilated = cv2.dilate(blue_mask.astype(np.uint8) * 255, blue_neighbor_kernel, iterations=1)
    
    # Green pixels that touch blue (blue-facing green) - these we want to KEEP
    blue_facing_green = green_mask_uint8 & blue_dilated
    
    # Erode the yellow-facing green to reduce thickness on the yellow side
    # But preserve blue-facing green completely
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    yellow_facing_eroded = cv2.erode(yellow_facing_green, erode_kernel, iterations=1)
    
    # Combine: keep all blue-facing green + eroded yellow-facing green
    # This preserves green on the outside edge while receding from yellow interior
    green_mask_balanced = blue_facing_green | yellow_facing_eroded
    
    # Also keep any green that doesn't touch either (isolated stroke segments)
    neither_facing = green_mask_uint8 & ~yellow_dilated & ~blue_dilated
    green_mask_balanced = green_mask_balanced | neither_facing
    
    # Apply morphological closing to fill gaps in green outlines
    # UPDATED: Reduced kernel from 7x7 to 5x5 and iterations from 2 to 1 to preserve thin outlines
    green_close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    green_mask_closed = cv2.morphologyEx(green_mask_balanced, cv2.MORPH_CLOSE, green_close_kernel, iterations=1)
    
    # Remove small disconnected "rogue" green pixels using connected component analysis
    # This keeps only large connected regions (the actual outlines) and removes isolated pixels
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(green_mask_closed, connectivity=8)
    
    # Calculate minimum area threshold (keep components > 25 pixels)
    # UPDATED: Reduced from 100 to 25 to preserve thinner green outline segments
    min_component_area = 25
    
    # Create cleaned mask keeping only significant components
    green_mask_clean = np.zeros_like(green_mask_closed)
    for i in range(1, num_labels):  # Skip background (label 0)
        if stats[i, cv2.CC_STAT_AREA] >= min_component_area:
            green_mask_clean[labels == i] = 255
    
    # SKIP Gaussian blur for green to prevent thickness stacking
    # The blur was fattening strokes; just use the cleaned mask directly
    green_mask_final = green_mask_clean > 0
    
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


def bilateral_smooth_edges(img, d=9, sigma_color=50, sigma_space=50):
    """
    Apply bilateral filter to smooth texture while preserving edges.
    UPDATED: Reduced parameters (d=9, sigma=50) to preserve thin outlines like green borders.
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


def morphological_cleanup(img, kernel_size=3, text_mask=None):
    """
    Apply gentle morphological operations to clean up noise.
    UPDATED: Reduced kernel size and iterations to preserve thin outlines.
    """
    print(f"Applying gentle morphological cleanup (kernel={kernel_size})...")
    
    # Use a smaller kernel to avoid eating thin lines
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # 1. Opening: Removes small white noise. 
    # Reduced to 1 iteration to prevent erasing thin outlines.
    opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 2. Closing: Fills small dark holes.
    cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # If text mask provided, blend original with cleaned to preserve text
    if text_mask is not None:
        print("  Applying text protection - preserving original in text regions...")
        text_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        text_cleaned = cv2.morphologyEx(img, cv2.MORPH_CLOSE, text_kernel, iterations=1)
        
        text_mask_3ch = cv2.cvtColor(text_mask, cv2.COLOR_GRAY2BGR)
        text_mask_float = text_mask_3ch.astype(np.float32) / 255.0
        
        cleaned = (text_mask_float * text_cleaned + (1 - text_mask_float) * cleaned).astype(np.uint8)
    
    return cleaned


def snap_to_palette(img, protect_outlines=False):
    """
    Snap every pixel to the nearest palette color using Euclidean distance.
    UPDATED: Can protect outline color pixels from being reassigned.
    """
    print("Snapping to palette colors...")
    
    # If protecting outlines, save original outline pixels
    outline_mask = None
    original_img = None
    if protect_outlines:
        original_img = img.copy()
        outline_colors = ['neon_green', 'outline_magenta', 'dark_purple', 'vibrant_red']
        outline_mask = np.zeros(img.shape[:2], dtype=bool)
        for color_name in outline_colors:
            color_bgr = np.array(PALETTE[color_name], dtype=np.uint8)
            outline_mask |= np.all(img == color_bgr, axis=2)
    
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
    
    # Restore protected outline pixels
    if protect_outlines and outline_mask is not None:
        quantized[outline_mask] = original_img[outline_mask]
    
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


def remove_isolated_specs(img, min_area=50):
    """
    Remove small isolated color specs that appear as noise.
    Uses connected component analysis per color channel to find and remove
    small disconnected regions that are likely artifacts.
    """
    print(f"Removing isolated specs (min_area={min_area})...")
    
    img_clean = img.copy()
    specs_removed = 0
    
    # Process each palette color
    for color_name, color_bgr in PALETTE.items():
        # Skip background colors and black
        if color_name in ['sky_blue', 'black']:
            continue
            
        color_bgr = np.array(color_bgr, dtype=np.uint8)
        
        # Find pixels of this color
        mask = np.all(img == color_bgr, axis=2).astype(np.uint8) * 255
        
        if np.sum(mask) == 0:
            continue
        
        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        # Find components that are too small (isolated specs)
        for i in range(1, num_labels):  # Skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area:
                # This is an isolated spec - replace with surrounding color
                spec_mask = (labels == i)
                
                # Find the most common neighboring color (excluding this color)
                # Dilate the spec mask to find neighbors
                dilated = cv2.dilate(spec_mask.astype(np.uint8) * 255, 
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
                neighbor_mask = (dilated > 0) & ~spec_mask
                
                if np.sum(neighbor_mask) > 0:
                    # Get neighboring pixels
                    neighbor_colors = img[neighbor_mask]
                    
                    # Find most common neighbor color
                    unique_colors, counts = np.unique(neighbor_colors, axis=0, return_counts=True)
                    most_common_idx = np.argmax(counts)
                    replacement_color = unique_colors[most_common_idx]
                    
                    # Replace the spec with neighbor color
                    img_clean[spec_mask] = replacement_color
                    specs_removed += 1
    
    print(f"  Removed {specs_removed} isolated specs")
    return img_clean


def apply_anti_aliasing(img):
    """
    Apply subtle anti-aliasing effect to color boundaries for smoother appearance.
    Uses guided filter to smooth transitions while maintaining sharp edges.
    """
    print("Applying anti-aliasing for smoother color transitions...")
    
    # Convert to float for processing
    img_float = img.astype(np.float32) / 255.0
    
    # Apply guided filter (edge-aware smoothing)
    # This smooths the image while preserving strong edges
    radius = 2
    eps = 0.01
    
    # Process each channel
    smoothed = np.zeros_like(img_float)
    for c in range(3):
        # Use the channel itself as the guide
        guide = img_float[:, :, c]
        src = img_float[:, :, c]
        
        # Box filter computations for guided filter
        mean_guide = cv2.boxFilter(guide, -1, (radius*2+1, radius*2+1))
        mean_src = cv2.boxFilter(src, -1, (radius*2+1, radius*2+1))
        corr_guide = cv2.boxFilter(guide * guide, -1, (radius*2+1, radius*2+1))
        corr_guide_src = cv2.boxFilter(guide * src, -1, (radius*2+1, radius*2+1))
        
        var_guide = corr_guide - mean_guide * mean_guide
        cov_guide_src = corr_guide_src - mean_guide * mean_src
        
        a = cov_guide_src / (var_guide + eps)
        b = mean_src - a * mean_guide
        
        mean_a = cv2.boxFilter(a, -1, (radius*2+1, radius*2+1))
        mean_b = cv2.boxFilter(b, -1, (radius*2+1, radius*2+1))
        
        smoothed[:, :, c] = mean_a * guide + mean_b
    
    # Convert back to uint8
    result = (smoothed * 255).clip(0, 255).astype(np.uint8)
    
    return result


def uniform_outline_width(img):
    """
    Normalize outline widths for consistent, professional appearance.
    Uses morphological operations to create uniform outline thickness.
    UPDATED: Skip neon_green entirely to prevent thickness stacking.
    """
    print("Normalizing outline widths for consistency...")
    
    img_result = img.copy()
    
    # Define outline colors - SKIP neon_green to prevent thickness stacking
    # neon_green is already thin and should stay thin
    outline_colors = {
        'outline_magenta': PALETTE['outline_magenta'],
        'dark_purple': PALETTE['dark_purple'],
        'vibrant_red': PALETTE['vibrant_red']
    }
    
    for color_name, color_bgr in outline_colors.items():
        color_bgr = np.array(color_bgr, dtype=np.uint8)
        
        # Find pixels of this outline color
        mask = np.all(img == color_bgr, axis=2).astype(np.uint8) * 255
        
        if np.sum(mask) < 1000:  # Skip if very few pixels
            continue
        
        # Apply morphological closing to fill small gaps in outlines
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)
        
        # For outline colors, only reinforce existing pixels - don't expand inward
        # This prevents green from flooding over yellow
        original_color_mask = np.all(img == color_bgr, axis=2)
        img_result[(mask_closed > 0) & original_color_mask] = color_bgr
    
    return img_result


def vectorize_edges(img, straightness_threshold=0.001, min_contour_area=500):
    """
    Create vector-like clean edges.
    UPDATED: Lower threshold (0.001) to preserve organic curves (hands, feet) 
    while still sharpening blocky edges.
    UPDATED: Treat outline colors (neon_green, outline_magenta, etc.) as STROKES,
    not filled regions, to preserve the outline-on-fill layering.
    """
    print("Vectorizing edges (High Fidelity Mode)...")
    
    img_result = img.copy()
    total_contours_processed = 0
    
    # Define outline colors that should be drawn as strokes, not fills
    outline_color_names = {
        'neon_green',
        'outline_magenta',
        'dark_purple',
        'vibrant_red'
    }
    
    # Process each palette color (except background and black)
    for color_name, color_bgr in PALETTE.items():
        if color_name in ['sky_blue', 'black']:
            continue
            
        color_bgr = np.array(color_bgr, dtype=np.uint8)
        mask = np.all(img == color_bgr, axis=2).astype(np.uint8) * 255
        
        if np.sum(mask) < min_contour_area:
            continue
        
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        
        new_mask = np.zeros_like(mask)
        is_outline_color = color_name in outline_color_names
        
        for i, contour in enumerate(contours):
            if cv2.contourArea(contour) < 50: 
                continue
            
            total_contours_processed += 1
            perimeter = cv2.arcLength(contour, True)
            
            # Check for rectangularity (Ladder rungs)
            epsilon_rough = 0.02 * perimeter
            approx_rough = cv2.approxPolyDP(contour, epsilon_rough, True)
            
            # Determine drawing mode: stroke for outline colors, fill for others
            if is_outline_color:
                # Draw as STROKE, not fill - preserves outline nature
                # UPDATED: Use thickness=1 for neon_green to match original thin accent look
                # Green is a halo/accent, not a structural border
                if color_name == 'neon_green':
                    thickness = 1  # Thin stroke for green accent
                else:
                    thickness = min(2, max(1, int(0.003 * perimeter)))  # Thin for other outlines
                epsilon = straightness_threshold * perimeter
                approx = cv2.approxPolyDP(contour, epsilon, True)
                cv2.drawContours(new_mask, [approx], -1, 255, thickness=thickness)
            elif len(approx_rough) == 4 and cv2.isContourConvex(approx_rough):
                # If it's a simple rectangle, enforce straight lines (filled)
                approx = _snap_to_right_angles(approx_rough)
                cv2.drawContours(new_mask, [approx], -1, 255, -1)
            else:
                # If it's a complex shape (Figures, Footprints), PRESERVE CURVES (filled)
                epsilon = straightness_threshold * perimeter 
                approx = cv2.approxPolyDP(contour, epsilon, True)
                cv2.drawContours(new_mask, [approx], -1, 255, -1)
        
        # Handle holes (hierarchy level 1) - only for filled regions
        if hierarchy is not None and not is_outline_color:
            for i, contour in enumerate(contours):
                if hierarchy[0][i][3] != -1: # It's a hole
                    if cv2.contourArea(contour) >= 50:
                        perimeter = cv2.arcLength(contour, True)
                        epsilon = 0.001 * perimeter # Keep holes precise
                        approx = cv2.approxPolyDP(contour, epsilon, True)
                        cv2.drawContours(new_mask, [approx], -1, 0, -1)
        
        img_result[new_mask > 0] = color_bgr
    
    print(f"  Processed {total_contours_processed} contours (High Fidelity)")
    return img_result


def _snap_to_right_angles(approx):
    """
    Snap a 4-point polygon's angles to 90 degrees if they're close.
    This creates perfectly rectangular shapes.
    """
    if len(approx) != 4:
        return approx
    
    points = approx.reshape(4, 2).astype(np.float32)
    
    # Calculate angles at each corner
    angles = []
    for i in range(4):
        p1 = points[(i - 1) % 4]
        p2 = points[i]
        p3 = points[(i + 1) % 4]
        
        v1 = p1 - p2
        v2 = p3 - p2
        
        # Calculate angle
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.arccos(cos_angle) * 180 / np.pi
        angles.append(angle)
    
    # Check if all angles are close to 90 degrees (within 15 degrees)
    all_right_angles = all(75 < angle < 105 for angle in angles)
    
    if all_right_angles:
        # Fit a minimum area rectangle to get perfectly straight edges
        rect = cv2.minAreaRect(approx)
        box = cv2.boxPoints(rect)
        return np.int0(box).reshape(-1, 1, 2)
    
    return approx


def smooth_jagged_edges(img):
    """
    Apply final edge smoothing to remove remaining jaggedness.
    Uses a combination of morphological operations and contour smoothing.
    UPDATED: Skip neon_green entirely to prevent thickness stacking.
    """
    print("Smoothing jagged edges for final cleanup...")
    
    img_result = img.copy()
    
    # Define outline colors that should be treated as strokes
    # SKIP neon_green entirely - it's already thin and any processing fattens it
    outline_colors_to_process = {
        'outline_magenta',
        'dark_purple',
        'vibrant_red'
    }
    
    # Process each color region
    for color_name, color_bgr in PALETTE.items():
        # Skip background, black, and neon_green (to prevent thickness stacking)
        if color_name in ['sky_blue', 'black', 'neon_green']:
            continue
            
        color_bgr = np.array(color_bgr, dtype=np.uint8)
        
        # Find pixels of this color
        mask = np.all(img == color_bgr, axis=2).astype(np.uint8) * 255
        
        if np.sum(mask) < 100:
            continue
        
        # Apply morphological smoothing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        if color_name in outline_colors_to_process:
            # OUTLINES: only light closing to smooth gaps, no opening
            # Opening erodes thin strokes which destroys outline topology
            smoothed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        else:
            # FILLS: opening + closing is safe for solid regions
            # Opening removes small protrusions (bumps outward)
            # Closing removes small intrusions (bumps inward)
            smoothed = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Apply Gaussian blur then threshold to smooth edges
        blurred = cv2.GaussianBlur(smoothed, (3, 3), 0)
        _, final_mask = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
        
        # Update result
        img_result[final_mask > 0] = color_bgr
    
    return img_result


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
    
    # Phase 4d: Remove isolated specs (white dots, color noise)
    img_despecked = remove_isolated_specs(img_quantized, min_area=50)
    
    # Phase 4e: Normalize outline widths for consistency
    img_uniform = uniform_outline_width(img_despecked)
    
    # Phase 4f: Vectorize edges for clean, vector-like appearance
    # Straightens rectangles while preserving curves/circles
    img_vectorized = vectorize_edges(img_uniform)
    
    # Phase 4g: Smooth jagged edges for final cleanup
    img_smooth_edges = smooth_jagged_edges(img_vectorized)
    
    # Phase 4h: Edge-preserving smooth to reduce any remaining jaggedness
    img_smooth_outlines = edge_preserving_smooth(img_smooth_edges)
    
    # Phase 4i: Apply anti-aliasing for smoother color transitions
    img_antialiased = apply_anti_aliasing(img_smooth_outlines)
    
    # Phase 4j: Solidify color regions (remove any remaining texture)
    img_solid = solidify_color_regions(img_antialiased)
    
    # Phase 5: Downscale back to original size
    print(f"Downscaling to original size: {original_size}")
    img_final = cv2.resize(img_solid, original_size, interpolation=cv2.INTER_AREA)
    
    # Phase 6: Final palette enforcement (protect outlines from being reassigned)
    img_final = snap_to_palette(img_final, protect_outlines=True)
    
    # Phase 7: Final spec removal at output resolution
    img_final = remove_isolated_specs(img_final, min_area=20)
    
    # Phase 8: Final edge smoothing at output resolution
    img_final = smooth_jagged_edges(img_final)
    
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
