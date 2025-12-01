# src/utils.py
"""
Utility functions for the document processing pipeline.
"""

import re
from typing import Optional


def normalize_text_basic(text: str) -> str:
    """
    Basic text normalization for OCR output.
    
    Args:
        text: Raw text string
    
    Returns:
        Normalized text
    """
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def normalize_text_full(text: str) -> str:
    """
    Full text normalization including Unicode and special character handling.
    
    Args:
        text: Raw text string
    
    Returns:
        Fully normalized text
    """
    if not text:
        return ""
    
    # Basic normalization
    text = normalize_text_basic(text)
    
    # Normalize Unicode characters
    import unicodedata
    text = unicodedata.normalize('NFKC', text)
    
    # Replace common problematic characters
    replacements = {
        '\u2018': "'",  # Left single quote
        '\u2019': "'",  # Right single quote
        '\u201c': '"',  # Left double quote
        '\u201d': '"',  # Right double quote
        '\u2013': '-',  # En dash
        '\u2014': '-',  # Em dash
        '\u2026': '...',  # Ellipsis
        '\u00a0': ' ',  # Non-breaking space
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text


def clean_filename(filename: str) -> str:
    """
    Clean a filename by removing invalid characters.
    
    Args:
        filename: Original filename
    
    Returns:
        Cleaned filename safe for file systems
    """
    # Remove invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Remove leading/trailing spaces and dots
    filename = filename.strip('. ')
    
    return filename


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union between two bounding boxes.
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    
    Returns:
        float: IoU value between 0 and 1
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def box_contains(outer_box, inner_box, margin=0):
    """
    Check if outer_box contains inner_box.
    
    Args:
        outer_box: [x1, y1, x2, y2]
        inner_box: [x1, y1, x2, y2]
        margin: Allowed margin of error
    
    Returns:
        bool: True if outer contains inner
    """
    return (outer_box[0] - margin <= inner_box[0] and
            outer_box[1] - margin <= inner_box[1] and
            outer_box[2] + margin >= inner_box[2] and
            outer_box[3] + margin >= inner_box[3])


def merge_boxes(boxes):
    """
    Merge a list of bounding boxes into a single encompassing box.
    
    Args:
        boxes: List of [x1, y1, x2, y2] boxes
    
    Returns:
        [x1, y1, x2, y2] merged box
    """
    if not boxes:
        return None
    
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    
    return [x1, y1, x2, y2]
