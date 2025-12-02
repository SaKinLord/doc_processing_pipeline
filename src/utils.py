# src/utils.py
"""
Utility functions for the document processing pipeline.
"""

import re
from typing import Optional, List, Tuple


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


# =============================================================================
# 🆕 ADDED: Missing utility functions for table extraction
# =============================================================================

def bbox_center(bbox) -> Tuple[float, float]:
    """
    Calculate the center point of a bounding box.
    
    This function is used by table_extraction.py for cell indexing
    and borderless table detection.
    
    Args:
        bbox: Bounding box as [x1, y1, x2, y2] or any 4-element sequence
    
    Returns:
        Tuple (cx, cy) - center x and y coordinates
    
    Examples:
        >>> bbox_center([0, 0, 100, 50])
        (50.0, 25.0)
        >>> bbox_center([10, 20, 30, 40])
        (20.0, 30.0)
    """
    if bbox is None:
        raise ValueError("bbox cannot be None")
    
    # Handle both list and dict formats
    if isinstance(bbox, dict):
        # Some systems use {'x1': ..., 'y1': ..., 'x2': ..., 'y2': ...}
        x1 = bbox.get('x1', bbox.get('left', 0))
        y1 = bbox.get('y1', bbox.get('top', 0))
        x2 = bbox.get('x2', bbox.get('right', 0))
        y2 = bbox.get('y2', bbox.get('bottom', 0))
    else:
        if len(bbox) != 4:
            raise ValueError(f"Expected bbox with 4 elements, got {len(bbox)}")
        x1, y1, x2, y2 = bbox
    
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    
    return (cx, cy)


def cluster_1d(values: List[float], tol: float = 10) -> List[float]:
    """
    Cluster 1D numeric values within a tolerance and return cluster centers.
    
    This is used for finding row/column positions in tables by grouping
    similar coordinates together.
    
    Args:
        values: List of numeric values to cluster
        tol: Tolerance for grouping values together (default: 10 pixels)
    
    Returns:
        List of cluster center values, sorted in ascending order
    
    Examples:
        >>> cluster_1d([10, 12, 50, 52, 100])
        [11.0, 51.0, 100.0]
        >>> cluster_1d([5, 5, 5, 100, 101, 102])
        [5.0, 101.0]
    """
    if not values:
        return []
    
    # Sort values
    sorted_vals = sorted(values)
    
    # Group values within tolerance using a simple greedy approach
    clusters = []
    current_cluster = [sorted_vals[0]]
    
    for val in sorted_vals[1:]:
        # Check if this value is close enough to the last value in current cluster
        if abs(val - current_cluster[-1]) <= tol:
            current_cluster.append(val)
        else:
            # Current cluster is complete, calculate its center
            cluster_center = sum(current_cluster) / len(current_cluster)
            clusters.append(cluster_center)
            # Start a new cluster
            current_cluster = [val]
    
    # Don't forget to add the last cluster
    if current_cluster:
        cluster_center = sum(current_cluster) / len(current_cluster)
        clusters.append(cluster_center)
    
    return sorted(clusters)


def boxes_overlap(box1, box2, threshold: float = 0.0) -> bool:
    """
    Check if two bounding boxes overlap.
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
        threshold: Minimum IoU to consider as overlap (default: any overlap)
    
    Returns:
        bool: True if boxes overlap
    """
    iou = calculate_iou(box1, box2)
    return iou > threshold


def expand_box(box, margin: int):
    """
    Expand a bounding box by a margin on all sides.
    
    Args:
        box: [x1, y1, x2, y2]
        margin: Pixels to expand on each side
    
    Returns:
        [x1, y1, x2, y2] expanded box
    """
    return [
        box[0] - margin,
        box[1] - margin,
        box[2] + margin,
        box[3] + margin
    ]


def clip_box_to_image(box, width: int, height: int):
    """
    Clip a bounding box to fit within image dimensions.
    
    Args:
        box: [x1, y1, x2, y2]
        width: Image width
        height: Image height
    
    Returns:
        [x1, y1, x2, y2] clipped box
    """
    return [
        max(0, min(box[0], width)),
        max(0, min(box[1], height)),
        max(0, min(box[2], width)),
        max(0, min(box[3], height))
    ]
