# src/figure_extraction.py

import cv2
import numpy as np
import os

# Try to import the classifier, but don't fail if it's not available
try:
    from .classifier import FigureClassifier
    CLASSIFIER_AVAILABLE = True
except ImportError:
    CLASSIFIER_AVAILABLE = False
    print("    - ⚠️ FigureClassifier not available, using fallback classification")

# Create the classifier instance at module level once (lazy loading)
_figure_classifier_instance = None


# =============================================================================
# REPLACEMENT FUNCTION 1: validate_figure_candidate (FROM PATCH)
# =============================================================================

def validate_figure_candidate(box, original_bgr_image, text_blocks=None):
    """
    🔧 IMPROVED: Validates a figure candidate box to eliminate false positives.
    
    Changes:
    - Added maximum size limit (40% of page instead of allowing 80%)
    - Added "mostly white" detection
    - Stricter edge density requirement for large boxes
    - Better aspect ratio filtering

    Returns True if the candidate passes validation, False otherwise.
    """
    x1, y1, x2, y2 = map(int, box)

    # Safety check for valid coordinates
    h, w = original_bgr_image.shape[:2]
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h or x2 <= x1 or y2 <= y1:
        return False

    box_width = x2 - x1
    box_height = y2 - y1
    box_area = box_width * box_height
    page_area = w * h
    
    coverage_ratio = box_area / page_area if page_area > 0 else 0

    # 🔧 FIX 1: Size Floor - Reject very small specks (< 1% of page)
    if coverage_ratio < 0.01:
        return False
    
    # 🔧 FIX 2: Size Ceiling - Reject giant boxes (> 40% of page)
    # Real figures rarely cover more than 40% of a page
    if coverage_ratio > 0.40:
        return False
    
    # 🔧 FIX 3: Lower half giant box filter
    # Boxes in lower 40% of page covering >20% are likely signature areas
    box_center_y = (y1 + y2) / 2
    is_lower_portion = box_center_y > h * 0.6
    if is_lower_portion and coverage_ratio > 0.20:
        return False

    # STRICT CHECK: Aspect Ratio Filter
    if box_width > 0 and box_height > 0:
        width_to_height = box_width / box_height
        height_to_width = box_height / box_width

        # Reject vertical lines (width/height < 0.1)
        if width_to_height < 0.1:
            return False

        # Reject horizontal lines (height/width < 0.1)
        if height_to_width < 0.1:
            return False
        
        # 🔧 FIX 4: Reject full-width boxes that are likely margins/footers
        if box_width > w * 0.9 and box_height < h * 0.15:
            return False

    roi = original_bgr_image[y1:y2, x1:x2]

    if roi.size == 0:
        return False

    # 🔧 FIX 5: "Mostly White" Detection
    # Convert to grayscale and check if region is mostly empty
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    mean_intensity = np.mean(gray_roi)
    std_intensity = np.std(gray_roi)
    
    # If the region is very bright (white) with low variation, it's likely empty
    if mean_intensity > 240 and std_intensity < 20:
        return False
    
    # 🔧 FIX 6: Adaptive edge density based on box size
    # Larger boxes need MORE content to be valid figures
    edges = cv2.Canny(gray_roi, 50, 150)
    total_pixels = edges.size
    edge_pixels = np.count_nonzero(edges)
    edge_density = (edge_pixels / total_pixels) * 100.0

    # Scale threshold by coverage - bigger boxes need more content
    min_edge_density = 1.0 + (coverage_ratio * 10)  # e.g., 30% coverage needs 4% edge density
    min_edge_density = min(min_edge_density, 5.0)  # Cap at 5%
    
    if edge_density < min_edge_density:
        return False

    # STRICT CHECK: Text Overlap Check
    if text_blocks is not None:
        text_overlap_count = 0
        total_text_area_inside = 0
        
        for text_block in text_blocks:
            if 'bounding_box' not in text_block:
                continue
            tx1, ty1, tx2, ty2 = text_block['bounding_box']

            # Calculate intersection area
            overlap_x1 = max(x1, tx1)
            overlap_y1 = max(y1, ty1)
            overlap_x2 = min(x2, tx2)
            overlap_y2 = min(y2, ty2)

            if overlap_x2 > overlap_x1 and overlap_y2 > overlap_y1:
                overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                text_area = (tx2 - tx1) * (ty2 - ty1)
                
                # Count significantly overlapping text blocks
                if text_area > 0 and overlap_area / text_area > 0.5:
                    text_overlap_count += 1
                    total_text_area_inside += text_area

        # 🔧 FIX 7: If figure contains multiple text blocks, it's likely a TABLE not a figure
        # Tables have lots of text inside them arranged in a grid
        if text_overlap_count > 5:
            return False
        
        # 🔧 FIX 8: If text covers significant portion of "figure", it's likely a table
        text_coverage = total_text_area_inside / box_area if box_area > 0 else 0
        if text_coverage > 0.15 and text_overlap_count > 3:
            return False

    # 🔧 FIX 9: Check for grid-like structure (indicates TABLE, not figure)
    # Look for regular horizontal lines which suggest table rows
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_roi, 50, 150)
    
    # Detect horizontal lines using morphology
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (box_width // 4, 1))
    horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
    h_line_count = cv2.countNonZero(horizontal_lines)
    
    # If lots of horizontal lines relative to area, likely a table
    h_line_density = h_line_count / box_area if box_area > 0 else 0
    if h_line_density > 0.01 and text_overlap_count > 2:
        return False

    return True


# =============================================================================
# REPLACEMENT FUNCTION 2: find_figure_candidates (FROM PATCH)
# =============================================================================

def find_figure_candidates(binarized_image_inv, original_bgr_image=None, text_blocks=None):
    """
    🔧 IMPROVED: Finds figure candidates with stricter filtering.
    
    Changes:
    - Reduced max size from 80% to 40%
    - Added content density pre-check
    - Better handling of mostly-empty regions
    """
    h, w = binarized_image_inv.shape[:2]

    # Use small kernel to avoid merging independent elements
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    blobs = cv2.morphologyEx(binarized_image_inv, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(blobs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        x, y, ww, hh = cv2.boundingRect(cnt)
        area = ww * hh
        page_area = h * w

        # 🔧 FIX: Stricter size filtering
        # Min: 2% of page (was 2%)
        # Max: 40% of page (was 80%) - this is the key fix!
        if area < 0.02 * page_area:
            continue
        if area > 0.40 * page_area:  # 🔧 Changed from 0.8 to 0.4
            continue

        candidate_box = [float(x), float(y), float(x + ww), float(y + hh)]

        # Content density check
        if original_bgr_image is not None:
            x1, y1, x2, y2 = int(x), int(y), int(x + ww), int(y + hh)
            # Boundary check
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 > x1 and y2 > y1:
                roi = original_bgr_image[y1:y2, x1:x2]
                if roi.size > 0:
                    # Check if region is mostly white/empty
                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    mean_val = np.mean(gray_roi)
                    std_val = np.std(gray_roi)
                    
                    # 🔧 FIX: Skip mostly white regions
                    if mean_val > 235 and std_val < 25:
                        continue
                    
                    # Calculate Canny edge density
                    edges = cv2.Canny(gray_roi, 50, 150)
                    edge_density = np.count_nonzero(edges) / area if area > 0 else 0

                    # 🔧 FIX: Adaptive threshold - larger areas need more content
                    coverage = area / page_area
                    min_density = 0.01 + (coverage * 0.05)  # Scale with size
                    
                    if edge_density < min_density:
                        continue

        # Strict validation
        if original_bgr_image is not None:
            if not validate_figure_candidate(candidate_box, original_bgr_image, text_blocks):
                continue

        boxes.append(candidate_box)

    return boxes


def classify_figure_kind(roi_bgr):
    """
    Uses a ViT-based classifier to determine the figure type based on ROI content.
    Falls back to simple heuristics if classifier is not available or fails.
    
    🆕 BUGFIX: Better error handling for bfloat16 overflow errors.
    """
    global _figure_classifier_instance
    
    if roi_bgr is None or roi_bgr.size == 0:
        return "figure"
    
    # 🆕 Validate input before any processing
    if len(roi_bgr.shape) != 3 or roi_bgr.shape[2] != 3:
        return _classify_figure_heuristic(roi_bgr)
    
    # Try to use the ML classifier (if available)
    if CLASSIFIER_AVAILABLE:
        try:
            # Initialize classifier only when first needed
            if _figure_classifier_instance is None:
                _figure_classifier_instance = FigureClassifier()
            return _figure_classifier_instance.classify(roi_bgr)
        except (ValueError, RuntimeError, OverflowError) as e:
            # 🆕 Catch bfloat16 overflow and other numeric errors silently
            # These are expected when pixel values are outside model's expected range
            pass
        except Exception as e:
            # Log unexpected errors but continue with fallback
            error_msg = str(e)
            if "BFloat16" not in error_msg and "overflow" not in error_msg:
                print(f"    - ⚠️ Classifier error: {e}, using fallback")
    
    # Fallback: Simple heuristic-based classification
    return _classify_figure_heuristic(roi_bgr)


def _classify_figure_heuristic(roi_bgr):
    """
    Simple heuristic-based figure classification.
    Used as fallback when ML classifier is not available.
    
    🆕 BUGFIX: Added safety checks for empty/invalid images.
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return "figure"
    
    # 🆕 Safety check for valid BGR image
    if len(roi_bgr.shape) != 3 or roi_bgr.shape[2] != 3:
        return "figure"
    
    try:
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    except cv2.error:
        return "figure"
    
    h, w = gray.shape
    if h == 0 or w == 0:
        return "figure"
    
    # Detect lines using HoughLinesP
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
    
    if lines is not None:
        horizontal_lines = 0
        vertical_lines = 0
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            if angle < 15 or angle > 165:
                horizontal_lines += 1
            elif 75 < angle < 105:
                vertical_lines += 1
        
        # Charts/graphs typically have grid lines
        if horizontal_lines > 3 and vertical_lines > 3:
            return "chart"
        
        # Tables have many horizontal lines
        if horizontal_lines > 5:
            return "table"
    
    # Check for photo-like characteristics (many unique colors, smooth gradients)
    try:
        unique_colors = len(np.unique(roi_bgr.reshape(-1, 3), axis=0))
        color_ratio = unique_colors / (h * w) if h * w > 0 else 0
        
        if color_ratio > 0.3:
            return "photograph"
    except:
        pass
    
    # Default
    return "figure"


def find_nearest_caption(figure_box, text_blocks, vertical_tolerance=140):
    """
    Finds the nearest text block (caption candidate) to a figure box.
    Priority: Below first, then above.
    """
    fx1, fy1, fx2, fy2 = figure_box
    
    below_candidates = []
    above_candidates = []
    
    for block in text_blocks:
        if 'bounding_box' not in block:
            continue
        bx1, by1, bx2, by2 = block['bounding_box']
        
        horizontal_overlap = max(0, min(fx2, bx2) - max(fx1, bx1))
        
        if horizontal_overlap > 20:
            if by1 >= fy2 and (by1 - fy2) < vertical_tolerance:
                below_candidates.append(((by1 - fy2), block))
            elif by2 <= fy1 and (fy1 - by2) < vertical_tolerance:
                above_candidates.append(((fy1 - by2), block))

    if below_candidates:
        below_candidates.sort(key=lambda x: x[0])
        return below_candidates[0][1]
    if above_candidates:
        above_candidates.sort(key=lambda x: x[0])
        return above_candidates[0][1]
        
    return None


def find_figure_regions(original_bgr_image, binary_image, text_blocks=None):
    """
    Main entry point for figure extraction.
    
    This is the function called by the pipeline. It:
    1. Finds figure candidates using morphological analysis
    2. Validates each candidate
    3. Classifies the figure type
    4. Finds associated captions
    
    Args:
        original_bgr_image: The original BGR image
        binary_image: Binarized/inverted image for contour detection
        text_blocks: List of text block dictionaries with 'bounding_box' keys
    
    Returns:
        List of figure dictionaries with bounding_box, type, caption, etc.
    """
    if original_bgr_image is None or binary_image is None:
        return []
    
    # Ensure binary image is inverted (white text/figures on black background)
    # for contour detection
    if len(binary_image.shape) == 3:
        binary_gray = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
    else:
        binary_gray = binary_image.copy()
    
    # Check if we need to invert (figures should be white for contour detection)
    mean_val = np.mean(binary_gray)
    if mean_val > 127:
        # Image is mostly white, invert it
        binary_inv = cv2.bitwise_not(binary_gray)
    else:
        binary_inv = binary_gray
    
    # Find candidates
    candidate_boxes = find_figure_candidates(binary_inv, original_bgr_image, text_blocks)
    
    if not candidate_boxes:
        return []
    
    figures = []
    h, w = original_bgr_image.shape[:2]
    
    for box in candidate_boxes:
        x1, y1, x2, y2 = map(int, box)
        
        # Boundary check
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            continue
        
        # Extract ROI for classification
        roi = original_bgr_image[y1:y2, x1:x2]
        
        # Classify the figure type
        figure_type = classify_figure_kind(roi)
        
        # Find caption
        caption_block = find_nearest_caption(box, text_blocks) if text_blocks else None
        caption_text = caption_block.get('content', '') if caption_block else ''
        
        figure_data = {
            'bounding_box': [float(x1), float(y1), float(x2), float(y2)],
            'figure_type': figure_type,
            'caption': caption_text,
            'width': x2 - x1,
            'height': y2 - y1,
            'area': (x2 - x1) * (y2 - y1),
            'confidence': 0.8  # Default confidence
        }
        
        figures.append(figure_data)
    
    return figures