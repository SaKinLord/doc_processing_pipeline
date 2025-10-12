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

def find_fields_by_location(text_blocks):
    """
    Separates text blocks into labels and values, and matches them more intelligently based on location.
    Priority: The nearest value in the same horizontal band and to the right.
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

        for value in values:
            if value["block_index"] in used_value_indices:
                continue
            
            vb = value["bounding_box"]
            
            # --- NEW AND STRICTER CHECKS ---
            
            # 1. Vertical Alignment Check: Is the value within the vertical band of the label?
            # The label's center point should be within the value's y boundaries (with a small tolerance).
            label_y_center = (lb[1] + lb[3]) / 2
            value_y_top = vb[1]
            value_y_bottom = vb[3]
            v_height = value_y_bottom - value_y_top
            
            is_vertically_aligned = (value_y_top - v_height * 0.5) < label_y_center < (value_y_bottom + v_height * 0.5)

            # 2. Horizontal Position Check: Is the value to the right of the label?
            is_to_the_right = vb[0] > lb[0]

            if is_vertically_aligned and is_to_the_right:
                # Horizontal gap between the two boxes
                dist = vb[0] - lb[2]
                
                # Is it at a reasonable distance? (If too far, it's unrelated)
                if 0 <= dist < 350 and dist < min_dist:
                    min_dist = dist
                    best_candidate = value
        
        if best_candidate:
            value_text = best_candidate["content"]
            key = label["key"]
            
            if key in NORMALIZERS:
                value_text = NORMALIZERS[key](value_text)
                
            extracted_fields[key] = {
                "value": value_text,
                "label_bbox": lb,
                "value_bbox": best_candidate["bounding_box"]
            }
            used_value_indices.add(best_candidate["block_index"])
            
    return extracted_fields