# src/figure_extraction.py

import cv2
import numpy as np
from .classifier import FigureClassifier # NEW: Import the classifier

# Create the classifier instance at module level once (lazy loading)
_figure_classifier_instance = None

def find_figure_candidates(binarized_image_inv):
    """Finds large and compact areas with low text density as figure candidates."""
    h, w = binarized_image_inv.shape[:2]
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    blobs = cv2.morphologyEx(binarized_image_inv, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(blobs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    for cnt in contours:
        x, y, ww, hh = cv2.boundingRect(cnt)
        area = ww * hh
        
        if area < 0.02 * h * w or area > 0.8 * h * w:
            continue
            
        boxes.append([float(x), float(y), float(x + ww), float(y + hh)])
        
    return boxes

# --- COMPLETELY RENEWED ---
def classify_figure_kind(roi_bgr):
    """
    Uses a ViT-based classifier to determine the figure type based on ROI content.
    """
    global _figure_classifier_instance
    
    # Initialize classifier only when first needed
    if _figure_classifier_instance is None:
        _figure_classifier_instance = FigureClassifier()

    if roi_bgr is None or roi_bgr.size == 0:
        return "figure"

    return _figure_classifier_instance.classify(roi_bgr)

def find_nearest_caption(figure_box, text_blocks, vertical_tolerance=140):
    """
    Finds the nearest text block (caption candidate) to a figure box.
    Priority: Below first, then above.
    """
    fx1, fy1, fx2, fy2 = figure_box
    
    below_candidates = []
    above_candidates = []
    
    for block in text_blocks:
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