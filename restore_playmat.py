#!/usr/bin/env python3
"""
Vinyl Playmat Digital Restoration Script
Removes wrinkles, glare, and texture from scanned vinyl playmat images
while preserving logos, text, stars, and silhouettes with accurate colors.
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

# Special color detection thresholds
NEAR_WHITE_THRESHOLD = 205  # Minimum channel value for near-white detection (catches RGB 205-255 range for light blue-white colors)
BLUE_GLARE_B_THRESHOLD = 230  # Blue channel threshold for glare detection
BLUE_GLARE_G_THRESHOLD = 180  # Green channel threshold for glare detection
BLUE_GLARE_R_THRESHOLD = 140  # Red channel threshold for glare detection
BLUE_GLARE_R_MAX = 205  # Red channel must be below this to avoid catching near-white colors

# Yellow edge gradient thresholds (for scanner edge artifacts)
# Bright yellow is BGR(1, 252, 253), so we look for high G and B, very low R
# Thresholds tuned to catch yellowish-green gradients at scan edges
NEAR_YELLOW_G_THRESHOLD = 180  # Green channel threshold for yellow detection
NEAR_YELLOW_B_THRESHOLD = 180  # Blue channel threshold for yellow detection
NEAR_YELLOW_R_MAX = 120  # Red channel must be below this for yellow

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


def detect_stars(img):
    """Detect star-shaped objects (5-pointed white polygons) including incomplete stars at edges."""
    print("Detecting stars...")
    
    # Create mask for bright white regions (lowered threshold to catch near-white)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, white_mask = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    star_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        # Stars should be medium-sized (not tiny noise, not huge logos)
        if 100 < area < 50000:
            # Approximate polygon
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
            
            # Check for star-like shape (more lenient: 5-14 vertices to catch incomplete stars)
            if 5 <= len(approx) <= 14:
                # Check convexity (stars are non-convex) - relaxed criteria
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                if hull_area > 0:
                    solidity = area / hull_area
                    # Stars have low solidity due to concave points - more lenient range
                    if 0.3 < solidity < 0.85:
                        cv2.drawContours(star_mask, [contour], -1, 255, -1)
    
    print(f"Star mask created: {np.sum(star_mask > 0)} pixels")
    return star_mask


def detect_text(img):
    """Detect small high-contrast text regions - improved to better preserve white text."""
    print("Detecting text...")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Create a mask for near-white regions (text is often white)
    _, white_text_mask = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
    
    # Also use adaptive threshold to detect high-contrast details
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # Combine both masks
    combined_binary = cv2.bitwise_or(binary, white_text_mask)
    
    # Find contours
    contours, _ = cv2.findContours(combined_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    text_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        # Text is typically small to medium sized (expanded range to catch more text)
        if 30 < area < 30000:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h) if h > 0 else 0
            
            # Text typically has certain aspect ratios (expanded range)
            if 0.08 < aspect_ratio < 15:
                # Expand bounding box more to ensure text protection
                padding = 8
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(img.shape[1], x + w + padding)
                y2 = min(img.shape[0], y + h + padding)
                text_mask[y1:y2, x1:x2] = 255
    
    print(f"Text mask created: {np.sum(text_mask > 0)} pixels")
    return text_mask


def detect_logo(img):
    """Detect the main logo (pink/white/purple sandwich)."""
    print("Detecting logo...")
    
    # Create mask for pink regions (hot_pink and dark_purple)
    pink_bgr = np.array(PALETTE['hot_pink'], dtype=np.uint8)
    purple_bgr = np.array(PALETTE['dark_purple'], dtype=np.uint8)
    
    # Define color ranges for pink and purple
    lower_pink = np.array([150, 0, 200], dtype=np.uint8)
    upper_pink = np.array([255, 100, 255], dtype=np.uint8)
    
    lower_purple = np.array([100, 0, 120], dtype=np.uint8)
    upper_purple = np.array([180, 50, 220], dtype=np.uint8)
    
    pink_mask = cv2.inRange(img, lower_pink, upper_pink)
    purple_mask = cv2.inRange(img, lower_purple, upper_purple)
    
    logo_mask = cv2.bitwise_or(pink_mask, purple_mask)
    
    # Dilate to connect logo parts
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    logo_mask = cv2.dilate(logo_mask, kernel, iterations=2)
    
    # Find large contiguous regions
    contours, _ = cv2.findContours(logo_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    final_logo_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for contour in contours:
        area = cv2.contourArea(contour)
        # Logo should be large
        if area > 50000:
            cv2.drawContours(final_logo_mask, [contour], -1, 255, -1)
    
    print(f"Logo mask created: {np.sum(final_logo_mask > 0)} pixels")
    return final_logo_mask


def create_protection_mask(img):
    """Create combined mask of all protected elements."""
    print("\n=== Phase 1: Creating Protection Masks ===")
    
    star_mask = detect_stars(img)
    text_mask = detect_text(img)
    logo_mask = detect_logo(img)
    
    # Combine all protection masks
    protection_mask = cv2.bitwise_or(star_mask, text_mask)
    protection_mask = cv2.bitwise_or(protection_mask, logo_mask)
    
    # Expand slightly to ensure protection
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    protection_mask = cv2.dilate(protection_mask, kernel, iterations=1)
    
    print(f"Total protected pixels: {np.sum(protection_mask > 0)}")
    return protection_mask


def detect_background_region(img, protection_mask):
    """Detect the sky blue background region."""
    print("\n=== Phase 2a: Detecting Background Region ===")
    
    sky_blue = np.array(PALETTE['sky_blue'], dtype=np.float32)
    
    # Convert to LAB for better color separation
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    img_float = img.astype(np.float32)
    
    # Calculate Euclidean distance to sky blue
    diff = img_float - sky_blue
    distance = np.sqrt(np.sum(diff ** 2, axis=2))
    
    # Create background mask (pixels close to sky blue)
    # Using a generous threshold to catch wrinkles and shadows
    bg_mask = (distance < 80).astype(np.uint8) * 255
    
    # Remove protected areas from background mask
    bg_mask = cv2.bitwise_and(bg_mask, cv2.bitwise_not(protection_mask))
    
    # Clean up the mask with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    print(f"Background pixels detected: {np.sum(bg_mask > 0)}")
    return bg_mask


def clean_background(img, bg_mask):
    """Replace background with flat sky blue, removing glare and wrinkles."""
    print("\n=== Phase 2b: Cleaning Background (The Background Nuke) ===")
    
    img_clean = img.copy()
    sky_blue = np.array(PALETTE['sky_blue'], dtype=np.uint8)
    
    # Replace all background pixels with flat sky blue
    img_clean[bg_mask > 0] = sky_blue
    
    print("Background replaced with flat sky_blue color")
    return img_clean


def preprocess_special_colors(img):
    """Pre-process special color ranges based on AI review of actual photo analysis.
    These ranges describe what EXISTS in the scanned photos (with texture/marbling),
    which will be replaced with flat, solid palette colors.
    """
    print("\n=== Pre-processing Special Colors ===")
    
    img_processed = img.copy()
    b, g, r = cv2.split(img)
    
    # Palette colors (flat, no texture)
    pure_white = np.array(PALETTE['pure_white'], dtype=np.uint8)
    sky_blue = np.array(PALETTE['sky_blue'], dtype=np.uint8)
    bright_yellow = np.array(PALETTE['bright_yellow'], dtype=np.uint8)
    hot_pink = np.array(PALETTE['hot_pink'], dtype=np.uint8)
    dark_purple = np.array(PALETTE['dark_purple'], dtype=np.uint8)
    vibrant_red = np.array(PALETTE['vibrant_red'], dtype=np.uint8)
    neon_green = np.array(PALETTE['neon_green'], dtype=np.uint8)
    black = np.array(PALETTE['black'], dtype=np.uint8)
    
    # 1. WHITE ELEMENTS (logos, stars, text)
    # Observed: RGB(205-250, 227-255, 245-255) - blue-biased from lighting
    # B consistently higher than R, never neutral white
    # BGR format: B=245-255, G=227-255, R=205-250
    white_mask = (b >= 245) & (b <= 255) & \
                 (g >= 227) & (g <= 255) & \
                 (r >= 205) & (r <= 250) & \
                 (b > r)  # B must be higher than R
    img_processed[white_mask] = pure_white
    white_count = np.sum(white_mask)
    print(f"  White elements: {white_count:,} pixels → pure_white")
    
    # Update channels
    b, g, r = cv2.split(img_processed)
    
    # 2. BLUE BACKGROUND - CLEAN (with marbling in source)
    # Observed: RGB(200-215, 220-235, 245-255) - base background
    # BGR format: B=245-255, G=220-235, R=200-215
    blue_clean_mask = (b >= 245) & (b <= 255) & \
                      (g >= 220) & (g <= 235) & \
                      (r >= 200) & (r <= 215)
    img_processed[blue_clean_mask] = sky_blue
    
    # 3. BLUE GLARE (folds & reflections)
    # Observed: RGB(160-185, 200-215, 235-255) - raised luminance, desaturated
    # BGR format: B=235-255, G=200-215, R=160-185
    blue_glare_mask = (b >= 235) & (b <= 255) & \
                      (g >= 200) & (g <= 215) & \
                      (r >= 160) & (r <= 185)
    img_processed[blue_glare_mask] = sky_blue
    
    # 4. BLUE DIRT/SCUFFING (removable noise)
    # Observed: RGB(120-135, 150-165, 180-200) - greyed, green-shifted
    # BGR format: B=180-200, G=150-165, R=120-135
    blue_dirt_mask = (b >= 180) & (b <= 200) & \
                     (g >= 150) & (g <= 165) & \
                     (r >= 120) & (r <= 135)
    img_processed[blue_dirt_mask] = sky_blue
    
    blue_total = np.sum(blue_clean_mask) + np.sum(blue_glare_mask) + np.sum(blue_dirt_mask)
    print(f"  Blue variations: {blue_total:,} pixels → sky_blue (flat)")
    
    # Update channels
    b, g, r = cv2.split(img_processed)
    
    # 5. YELLOW SILHOUETTES (figures)
    # Observed: RGB(195-255, 195-255, 0-20) - dense ink, slightly green-biased
    # BGR format: B=0-20, G=195-255, R=195-255
    yellow_silhouette_mask = (b >= 0) & (b <= 20) & \
                             (g >= 195) & (g <= 255) & \
                             (r >= 195) & (r <= 255)
    img_processed[yellow_silhouette_mask] = bright_yellow
    
    # 6. YELLOW BLOCK TEXT (brighter, cleaner)
    # Observed: RGB(240-255, 228-255, 0-15) - higher luminance
    # BGR format: B=0-15, G=228-255, R=240-255
    yellow_text_mask = (b >= 0) & (b <= 15) & \
                       (g >= 228) & (g <= 255) & \
                       (r >= 240) & (r <= 255)
    img_processed[yellow_text_mask] = bright_yellow
    
    # 7. YELLOW GLARE (specular highlights)
    # Observed: RGB(250-255, 250-255, 15-35) - slight B lift
    # BGR format: B=15-35, G=250-255, R=250-255
    yellow_glare_mask = (b >= 15) & (b <= 35) & \
                        (g >= 250) & (g <= 255) & \
                        (r >= 250) & (r <= 255)
    img_processed[yellow_glare_mask] = bright_yellow
    
    yellow_total = np.sum(yellow_silhouette_mask) + np.sum(yellow_text_mask) + np.sum(yellow_glare_mask)
    print(f"  Yellow variations: {yellow_total:,} pixels → bright_yellow (flat)")
    
    # Update channels
    b, g, r = cv2.split(img_processed)
    
    # 8. PINK MAIN FILL (foot graphic)
    # Observed: RGB(245-255, 0-10, 195-215) - saturated magenta-pink, stable
    # BGR format: B=195-215, G=0-10, R=245-255
    pink_main_mask = (b >= 195) & (b <= 215) & \
                     (g >= 0) & (g <= 10) & \
                     (r >= 245) & (r <= 255)
    img_processed[pink_main_mask] = hot_pink
    
    # 9. PINK BRIGHT OUTLINE
    # Observed: RGB(235-255, 0-20, 185-210) - same hue, slightly darker/thinner
    # BGR format: B=185-210, G=0-20, R=235-255
    pink_bright_mask = (b >= 185) & (b <= 210) & \
                       (g >= 0) & (g <= 20) & \
                       (r >= 235) & (r <= 255)
    img_processed[pink_bright_mask] = hot_pink
    
    pink_total = np.sum(pink_main_mask) + np.sum(pink_bright_mask)
    print(f"  Hot pink variations: {pink_total:,} pixels → hot_pink (flat)")
    
    # Update channels
    b, g, r = cv2.split(img_processed)
    
    # 10. DARKER PINK OUTLINE (adjacent, intentional shadow/edge)
    # Observed: RGB(200-235, 0-25, 160-190)
    # BGR format: B=160-190, G=0-25, R=200-235
    dark_pink_mask = (b >= 160) & (b <= 190) & \
                     (g >= 0) & (g <= 25) & \
                     (r >= 200) & (r <= 235)
    img_processed[dark_pink_mask] = dark_purple
    purple_count = np.sum(dark_pink_mask)
    print(f"  Dark purple outline: {purple_count:,} pixels → dark_purple (flat)")
    
    # Update channels
    b, g, r = cv2.split(img_processed)
    
    # 11. RED BLOCK OUTLINE (near-pure red)
    # Observed: RGB(250-255, 0-10, 0-10) - minor camera noise
    # BGR format: B=0-10, G=0-10, R=250-255
    red_mask = (b >= 0) & (b <= 10) & \
               (g >= 0) & (g <= 10) & \
               (r >= 250) & (r <= 255)
    img_processed[red_mask] = vibrant_red
    red_count = np.sum(red_mask)
    print(f"  Vibrant red outline: {red_count:,} pixels → vibrant_red (flat)")
    
    # Update channels
    b, g, r = cv2.split(img_processed)
    
    # 12. LIME GREEN OUTLINE (silhouette edge ONLY, thin high-contrast)
    # Observed: RGB(155-200, 180-215, 0-35) - never appears as fill
    # BGR format: B=0-35, G=180-215, R=155-200
    # CRITICAL: Only apply to actual edge pixels, not fills
    
    # First, detect potential green pixels by color
    green_color_mask = (b >= 0) & (b <= 35) & \
                       (g >= 180) & (g <= 215) & \
                       (r >= 155) & (r <= 200) & \
                       (g > b) & (g > r)  # G must be highest
    
    # Now detect edges in the image to ensure we only mark actual outline pixels
    gray = cv2.cvtColor(img_processed, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # Dilate edges slightly to capture thin outlines
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edge_zone = cv2.dilate(edges, kernel, iterations=2)
    
    # Green ONLY where both conditions are met: color matches AND it's on an edge
    green_mask = green_color_mask & (edge_zone > 0)
    
    img_processed[green_mask] = neon_green
    green_count = np.sum(green_mask)
    print(f"  Neon green outline (edges only): {green_count:,} pixels → neon_green (flat)")
    
    # Any remaining green-colored pixels that are NOT on edges should be yellow
    green_interior = green_color_mask & (edge_zone == 0)
    if np.sum(green_interior) > 0:
        img_processed[green_interior] = bright_yellow
        print(f"  Green-tinted interior (converted): {np.sum(green_interior):,} pixels → bright_yellow (flat)")
    
    # Update channels
    b, g, r = cv2.split(img_processed)
    
    # 13. DEADSPACE/VOID (scanner bed/background)
    # Observed: RGB(0-10, 0-10, 0-10)
    # BGR format: B=0-10, G=0-10, R=0-10
    black_mask = (b >= 0) & (b <= 10) & \
                 (g >= 0) & (g <= 10) & \
                 (r >= 0) & (r <= 10)
    img_processed[black_mask] = black
    black_count = np.sum(black_mask)
    print(f"  Deadspace/void: {black_count:,} pixels → black (flat)")
    
    print("  All texture and marbling will be removed by subsequent solidification phase")
    return img_processed


def snap_to_palette(img, protection_mask):
    """Snap all colors to the nearest palette color using vectorized operations."""
    print("\n=== Phase 3: Color Quantization ===")
    
    # Pre-process special color ranges
    img = preprocess_special_colors(img)
    
    img_float = img.astype(np.float32)
    h, w = img.shape[:2]
    
    # Reshape image to (n_pixels, 3)
    pixels = img_float.reshape(-1, 3)
    
    # Calculate distances to all palette colors (vectorized)
    # Shape: (n_pixels, n_colors)
    distances = np.sqrt(np.sum((pixels[:, np.newaxis, :] - PALETTE_ARRAY[np.newaxis, :, :]) ** 2, axis=2))
    
    # Find nearest color for each pixel
    nearest_idx = np.argmin(distances, axis=1)
    
    # Map to palette colors
    snapped = PALETTE_ARRAY[nearest_idx]
    
    # Reshape back to image
    result = snapped.reshape(h, w, 3).astype(np.uint8)
    
    print("All pixels snapped to palette colors")
    return result


def reinforce_outlines(img):
    """Reinforce silhouette outlines and logo borders."""
    print("\n=== Phase 3b: Reinforcing Outlines ===")
    
    # Apply bilateral filter to preserve edges while smoothing
    img_smooth = cv2.bilateralFilter(img, 9, 75, 75)
    
    # Detect edges
    gray = cv2.cvtColor(img_smooth, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # Dilate edges slightly to reinforce them
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    
    # Keep original colors at edge locations
    result = img.copy()
    result[edges_dilated > 0] = img_smooth[edges_dilated > 0]
    
    print("Outlines reinforced")
    return result


def fill_holes(img):
    """Fill single-pixel holes inside solid color regions."""
    print("\n=== Phase 3c: Filling Holes ===")
    
    # Use morphological closing to fill small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    # Split into channels and close each separately
    b, g, r = cv2.split(img)
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, kernel, iterations=1)
    g = cv2.morphologyEx(g, cv2.MORPH_CLOSE, kernel, iterations=1)
    r = cv2.morphologyEx(r, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    result = cv2.merge([b, g, r])
    print("Holes filled")
    return result


def solidify_color_regions(img):
    """Solidify color regions to remove scanner edge gradients and shading artifacts.
    Uses moderate median filtering to flatten regions while protecting small features."""
    print("\n=== Phase 3d: Solidifying Color Regions ===")
    
    result = img.copy()
    
    # Use MODERATE median filter to avoid destroying small features
    # Kernel size 5: Balances texture removal with detail preservation
    # Protects stars, thin text, and outlines from being blurred into adjacent colors
    kernel_size = 5
    img_median = cv2.medianBlur(img, kernel_size)
    
    # Minimum region size threshold: 500 pixels (lowered to catch smaller regions)
    MIN_REGION_SIZE = 500
    
    # For each major palette color, find its regions and apply median filter
    # This removes gradients within solid color areas while preserving edges
    # SKIP WHITE to protect stars, text, and logos from bleeding into blue background
    for color_name, color_bgr in PALETTE.items():
        # Skip black (borders/edges) and white (stars/text/logos)
        if color_name in ['black', 'pure_white']:
            continue
        
        color_arr = np.array(color_bgr, dtype=np.uint8)
        
        # Create mask for this color (exact match)
        color_mask = np.all(img == color_arr, axis=2).astype(np.uint8)
        
        if np.sum(color_mask) > MIN_REGION_SIZE:
            # Replace pixels of this color with median-filtered version
            # This removes gradients but the median filter preserves edges better than blur
            result[color_mask > 0] = img_median[color_mask > 0]
    
    print("Color regions solidified while protecting white elements")
    return result


def apply_edge_antialiasing(img):
    """Apply slight anti-aliasing to color boundaries."""
    print("\n=== Phase 4: Applying Edge Anti-Aliasing ===")
    
    # Detect edges
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # Dilate edges to create a zone for anti-aliasing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edge_zone = cv2.dilate(edges, kernel, iterations=2)
    
    # Apply gentle Gaussian blur
    img_blurred = cv2.GaussianBlur(img, (3, 3), 0.5)
    
    # Blend original and blurred at edges
    result = img.copy()
    alpha = 0.6  # Blend factor
    
    mask_3ch = cv2.cvtColor(edge_zone, cv2.COLOR_GRAY2BGR) / 255.0
    result = (result * (1 - alpha * mask_3ch) + img_blurred * alpha * mask_3ch).astype(np.uint8)
    
    print("Anti-aliasing applied to edges")
    return result


def force_exact_palette_colors(img):
    """Final cleanup: Force every pixel to be an exact palette color.
    Removes any salt-and-pepper noise, speckles, or intermediate colors
    created by filters/anti-aliasing to achieve perfectly clean vector-style output.
    Preserves neon_green outlines while preventing green fills."""
    print("\n=== Phase 5: Final Palette Enforcement ===")
    
    img_float = img.astype(np.float32)
    h, w = img.shape[:2]
    
    # Reshape image to (n_pixels, 3)
    pixels = img_float.reshape(-1, 3)
    
    # Calculate distances to all palette colors (vectorized)
    # Shape: (n_pixels, n_colors)
    distances = np.sqrt(np.sum((pixels[:, np.newaxis, :] - PALETTE_ARRAY[np.newaxis, :, :]) ** 2, axis=2))
    
    # Find nearest color for each pixel
    nearest_idx = np.argmin(distances, axis=1)
    
    # Map to exact palette colors
    snapped = PALETTE_ARRAY[nearest_idx]
    
    # Reshape back to image
    result = snapped.reshape(h, w, 3).astype(np.uint8)
    
    # Count pixels per color for verification
    unique_colors = {}
    for color_name, color_bgr in PALETTE.items():
        color_arr = np.array(color_bgr, dtype=np.uint8)
        mask = np.all(result == color_arr, axis=2)
        count = np.sum(mask)
        if count > 0:
            unique_colors[color_name] = count
    
    total_pixels = h * w
    exact_match = sum(unique_colors.values())
    exact_pct = (exact_match / total_pixels) * 100 if total_pixels > 0 else 0
    
    print(f"  100% exact palette colors ({exact_match:,} / {total_pixels:,} pixels)")
    print(f"  Colors present: {len(unique_colors)}/{len(PALETTE)}")
    for color_name, count in sorted(unique_colors.items(), key=lambda x: x[1], reverse=True):
        pct = (count / total_pixels) * 100
        print(f"    {color_name}: {count:,} pixels ({pct:.2f}%)")
    
    return result
    
    print(f"  Final palette distribution:")
    for name, count in sorted(unique_colors.items(), key=lambda x: x[1], reverse=True):
        pct = (count / total_pixels) * 100
        print(f"    {name:15}: {count:10,} pixels ({pct:5.2f}%)")
    print(f"  Total: {exact_match:,} / {total_pixels:,} pixels ({exact_pct:.1f}%) - perfectly flat")
    
    return result


def process_image(input_path, output_path=None):
    """Main processing pipeline."""
    print(f"\n{'='*60}")
    print(f"Processing: {input_path}")
    print(f"{'='*60}")
    
    # Load and upscale
    img, original_size = load_and_upscale(input_path, scale=3)
    
    # Phase 1: Protection Masking
    protection_mask = create_protection_mask(img)
    
    # Phase 2: Background Cleaning
    bg_mask = detect_background_region(img, protection_mask)
    img_clean = clean_background(img, bg_mask)
    
    # Apply bilateral filter to preserve edges before color snapping
    print("\n=== Applying Edge-Preserving Smoothing ===")
    img_smooth = cv2.bilateralFilter(img_clean, 9, 75, 75)
    
    # Phase 3: Color Quantization
    img_snapped = snap_to_palette(img_smooth, protection_mask)
    
    # Phase 3: Outline Reinforcement and Hole Filling
    img_reinforced = reinforce_outlines(img_snapped)
    img_filled = fill_holes(img_reinforced)
    img_solidified = solidify_color_regions(img_filled)
    
    # Phase 5: Force exact palette colors (removes all noise/speckles)
    img_final = force_exact_palette_colors(img_solidified)
    
    # Downscale back to original size
    print(f"\n=== Downscaling back to original size: {original_size} ===")
    img_output = cv2.resize(img_final, original_size, interpolation=cv2.INTER_AREA)
    
    # Final snap after downscaling to ensure no intermediate colors from resizing
    img_output = force_exact_palette_colors(img_output)
    
    # Generate output path if not provided
    if output_path is None:
        input_path_obj = Path(input_path)
        output_path = input_path_obj.parent / f"{input_path_obj.stem}_restored.png"
    
    # Save as PNG (lossless)
    cv2.imwrite(str(output_path), img_output, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    print(f"\n{'='*60}")
    print(f"Output saved to: {output_path}")
    print(f"{'='*60}\n")
    
    return img_output


def process_directory(input_dir, output_dir=None):
    """Process all JPG images in a directory."""
    input_path = Path(input_dir)
    
    if output_dir is None:
        output_dir = input_path / "restored"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True)
    
    # Find all JPG files
    jpg_files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.JPG"))
    
    if not jpg_files:
        print(f"No JPG files found in {input_dir}")
        return
    
    print(f"\nFound {len(jpg_files)} images to process")
    print(f"Output directory: {output_dir}\n")
    
    for i, jpg_file in enumerate(jpg_files, 1):
        print(f"\n[{i}/{len(jpg_files)}] Processing {jpg_file.name}...")
        try:
            output_path = output_dir / f"{jpg_file.stem}_restored.png"
            process_image(str(jpg_file), str(output_path))
        except Exception as e:
            print(f"ERROR processing {jpg_file.name}: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main entry point."""
    print("="*60)
    print("Vinyl Playmat Digital Restoration Script")
    print("="*60)
    
    if len(sys.argv) < 2:
        # Process current directory if no argument provided
        input_path = "."
        print("\nNo input provided, processing current directory...")
    else:
        input_path = sys.argv[1]
    
    # Check if input is a file or directory
    path_obj = Path(input_path)
    
    if not path_obj.exists():
        print(f"ERROR: Path does not exist: {input_path}")
        sys.exit(1)
    
    if path_obj.is_file():
        # Process single file
        output_path = None
        if len(sys.argv) >= 3:
            output_path = sys.argv[2]
        process_image(str(path_obj), output_path)
    else:
        # Process directory
        output_dir = None
        if len(sys.argv) >= 3:
            output_dir = sys.argv[2]
        process_directory(str(path_obj), output_dir)
    
    print("\n" + "="*60)
    print("Processing complete!")
    print("="*60)


if __name__ == "__main__":
    main()
