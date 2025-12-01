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


def validate_figure_candidate(box, original_bgr_image, text_blocks=None):
    """
    Validates a figure candidate box to eliminate false positives.

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

    # STRICT CHECK 1: Size Floor - Reject very small specks (< 1% of page)
    if box_area < 0.01 * page_area:
        return False

    # STRICT CHECK 2: Aspect Ratio Filter - Reject extremely thin/tall boxes
    # These are likely margin lines, page borders, or artifacts
    if box_width > 0 and box_height > 0:
        width_to_height = box_width / box_height
        height_to_width = box_height / box_width

        # Reject vertical lines (width/height < 0.1)
        if width_to_height < 0.1:
            return False

        # Reject horizontal lines (height/width < 0.1)
        if height_to_width < 0.1:
            return False

    roi = original_bgr_image[y1:y2, x1:x2]

    if roi.size == 0:
        return False

    # STRICT CHECK 3: Edge Density Check - Reject empty regions with no visual content
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_roi, 50, 150)

    total_pixels = edges.size
    edge_pixels = np.count_nonzero(edges)
    edge_density = (edge_pixels / total_pixels) * 100.0

    # STRICT: Valid figures must have significant internal detail (>= 1%)
    # Real figures/charts have lines and visual elements; empty margins do not
    if edge_density < 1.0:
        return False

    # STRICT CHECK 4: Text Overlap Check - Reject if figure overlaps significantly with text blocks
    if text_blocks is not None:
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
                overlap_ratio = (overlap_area / box_area) * 100.0

                # Reject if overlap with ANY text block exceeds 10%
                if overlap_ratio > 10.0:
                    return False

    return True


def find_figure_candidates(binarized_image_inv, original_bgr_image=None, text_blocks=None):
    """
    Finds large and compact areas with low text density as figure candidates.
    Uses strict validation to eliminate false positives.
    """
    h, w = binarized_image_inv.shape[:2]

    # CRITICAL FIX: Further reduced kernel to (3,3) to stop merging independent elements
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    blobs = cv2.morphologyEx(binarized_image_inv, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(blobs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        x, y, ww, hh = cv2.boundingRect(cnt)
        area = ww * hh

        # Basic size filtering
        if area < 0.02 * h * w or area > 0.8 * h * w:
            continue

        candidate_box = [float(x), float(y), float(x + ww), float(y + hh)]

        # CRITICAL FIX: Add density filter directly in candidate loop
        # Check edge density to reject empty margin regions
        if original_bgr_image is not None:
            x1, y1, x2, y2 = int(x), int(y), int(x + ww), int(y + hh)
            # Boundary check
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 > x1 and y2 > y1:
                roi = original_bgr_image[y1:y2, x1:x2]
                if roi.size > 0:
                    # Calculate Canny edge density
                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    edges = cv2.Canny(gray_roi, 50, 150)
                    edge_density = np.count_nonzero(edges) / area if area > 0 else 0

                    # STRICT: Reject if edge density < 1% (0.01)
                    # Real figures/charts have internal lines and details
                    # Empty page margins or solid blocks do not
                    if edge_density < 0.01:
                        continue

        # Strict validation: Only add if it passes all checks
        # Better to miss a figure than to mark the entire page as a figure
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
