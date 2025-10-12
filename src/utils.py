# src/utils.py

import re
import numpy as np
from difflib import SequenceMatcher

def bbox_center(box):
    """Returns the center coordinates (cx, cy) of a bounding box."""
    x1, y1, x2, y2 = map(float, box)
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def cluster_1d(vals, tol=6):
    """Clusters 1D values by tolerance and returns cluster centers."""
    if not vals:
        return []
    vals = sorted(vals)
    groups = [[vals[0]]]
    for v in vals[1:]:
        if abs(v - groups[-1][-1]) <= tol:
            groups[-1].append(v)
        else:
            groups.append([v])
    return [int(round(np.median(g))) for g in groups]

def iou(box_a, box_b):
    """Calculates the Intersection over Union (IoU) value between two bounding boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    
    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    
    intersection = inter_w * inter_h
    if intersection == 0:
        return 0.0
        
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - intersection
    
    return intersection / union if union > 0 else 0.0

def normalize_text_basic(text):
    """Cleans excess whitespace from text."""
    if not isinstance(text, str):
        return ""
    return re.sub(r'\s+', ' ', text).strip()

def similarity_ratio(a, b):
    """Calculates the similarity ratio between two texts."""
    a_norm = normalize_text_basic(a)
    b_norm = normalize_text_basic(b)
    return SequenceMatcher(None, a_norm, b_norm).ratio()