# src/field_extraction.py

import re
from datetime import datetime

# Keyword and regex patterns
FIELD_PATTERNS = {
    "fax": r"\bfax(?:\s*(?:no|number))?\b",
    "phone": r"\bphone(?:\s*(?:no|number))?\b",
    "date": r"\b(date|tarih)\b",
    "to": r"\b(to|recipient|alıcı)\b",
    "from": r"\b(from|gönderen)\b",
    "subject": r"\b(subject|konu)\b",
    "pages": r"\b(no\.?\s*of\s*pages|pages|sayfa)\b",
}

def _normalize_date(text):
    """Attempts to convert date formats in text to YYYY-MM-DD."""
    text = (text or "").strip()
    formats = ["%m/%d/%y", "%m/%d/%Y", "%m-%d-%y", "%m-%d-%Y", "%b %d %Y", "%b %d, %Y"]
    for fmt in formats:
        try:
            dt = datetime.strptime(text, fmt)
            year = dt.year if dt.year > 1000 else (dt.year + 2000 if dt.year < 50 else dt.year + 1900)
            return f"{year:04d}-{dt.month:02d}-{dt.day:02d}"
        except ValueError:
            pass
    return text

def _normalize_phone(text):
    """Converts phone number in text to international format (+12223334444)."""
    digits = re.sub(r'[^\d+]', '', (text or ""))
    if digits.startswith('+'):
        return digits
    if len(digits) == 10: # US assumption
        return f"+1{digits}"
    if len(digits) == 11 and digits.startswith('1'):
        return f"+{digits}"
    return f"+{digits}" if digits else text

NORMALIZERS = {
    "date": _normalize_date,
    "phone": _normalize_phone,
    "fax": _normalize_phone,
}

def _calculate_distance(label_bbox, value_bbox, direction):
    """
    Calculates the distance between label and value based on direction.
    Returns (distance, is_valid).
    """
    lb_x1, lb_y1, lb_x2, lb_y2 = label_bbox
    vb_x1, vb_y1, vb_x2, vb_y2 = value_bbox
    
    label_y_center = (lb_y1 + lb_y2) / 2
    value_y_center = (vb_y1 + vb_y2) / 2
    label_x_center = (lb_x1 + lb_x2) / 2
    value_x_center = (vb_x1 + vb_x2) / 2
    
    v_height = vb_y2 - vb_y1
    v_width = vb_x2 - vb_x1
    l_height = lb_y2 - lb_y1
    l_width = lb_x2 - lb_x1
    
    if direction == "right":
        # Value is to the right of label
        is_to_right = vb_x1 > lb_x1
        is_vertically_aligned = (vb_y1 - v_height * 0.5) < label_y_center < (vb_y2 + v_height * 0.5)
        
        if is_to_right and is_vertically_aligned:
            horizontal_gap = vb_x1 - lb_x2
            if 0 <= horizontal_gap < 350:
                return horizontal_gap, True
    
    elif direction == "below":
        # Value is below label
        is_below = vb_y1 > lb_y2
        # Check horizontal alignment (value should be somewhat aligned horizontally with label)
        is_horizontally_aligned = (vb_x1 < lb_x2 + l_width * 0.3) and (vb_x2 > lb_x1 - l_width * 0.3)
        
        if is_below and is_horizontally_aligned:
            vertical_gap = vb_y1 - lb_y2
            if 0 <= vertical_gap < 100:  # Max 100px below
                return vertical_gap, True
    
    elif direction == "right_below":
        # Value is to the right and slightly below (diagonal)
        is_to_right = vb_x1 > lb_x1
        is_slightly_below = 0 <= (vb_y1 - lb_y2) < 50
        
        if is_to_right and is_slightly_below:
            # Calculate Euclidean distance
            dx = max(0, vb_x1 - lb_x2)
            dy = max(0, vb_y1 - lb_y2)
            distance = (dx**2 + dy**2) ** 0.5
            if distance < 200:
                return distance, True
    
    return float('inf'), False

def find_fields_by_location(text_blocks):
    """
    ENHANCED: Separates text blocks into labels and values, and matches them intelligently
    based on location. Searches in multiple directions: right, below, and right-below.
    Priority: Nearest value in any valid direction.
    """
    labels = []
    values = []

    # Separate blocks into labels and values
    for i, block in enumerate(text_blocks):
        text = block.get("content", "")
        if not text:
            continue
        
        is_label = False
        for key, pattern in FIELD_PATTERNS.items():
            # The label is more likely to be standalone or at the beginning.
            match = re.search(pattern, text, re.IGNORECASE)
            if match and match.start() < 10: # If label is at the start of text
                labels.append({"key": key, "block_index": i, **block})
                is_label = True
                break
        if not is_label:
            values.append({"block_index": i, **block})

    extracted_fields = {}
    used_value_indices = set()

    for label in labels:
        if label["key"] in extracted_fields: # If there are multiple of the same label, take the first one found
            continue

        lb = label["bounding_box"]
        
        best_candidate = None
        min_dist = float('inf')
        best_direction = None

        for value in values:
            if value["block_index"] in used_value_indices:
                continue
            
            vb = value["bounding_box"]
            
            # Try all three directions and pick the closest valid one
            for direction in ["right", "below", "right_below"]:
                dist, is_valid = _calculate_distance(lb, vb, direction)
                
                if is_valid and dist < min_dist:
                    min_dist = dist
                    best_candidate = value
                    best_direction = direction
        
        if best_candidate:
            value_text = best_candidate["content"]
            key = label["key"]
            
            if key in NORMALIZERS:
                value_text = NORMALIZERS[key](value_text)
                
            extracted_fields[key] = {
                "value": value_text,
                "label_bbox": lb,
                "value_bbox": best_candidate["bounding_box"],
                "match_direction": best_direction  # NEW: Track which direction matched
            }
            used_value_indices.add(best_candidate["block_index"])
            
    return extracted_fields