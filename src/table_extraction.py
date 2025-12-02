# src/table_extraction.py
# 🔧 PATCHED VERSION with fixes for:
# - Wrong region detection (headers/footers instead of data tables)
# - OCR garbage in table cells
# - Blank forms detected as tables
# - Footer disclaimers captured as tables

import cv2
import numpy as np
import pandas as pd
import re
from . import utils


# =============================================================================
# 🔧 NEW: CONTENT VALIDATION FUNCTIONS
# =============================================================================

def is_likely_table_content(text: str) -> bool:
    """
    Check if text looks like actual table data (not garbage or headers).
    
    Good table content has:
    - Recognizable words or numbers
    - Names, dates, amounts, codes
    - Consistent patterns
    
    Bad content has:
    - Mostly non-alphanumeric characters
    - Random character sequences
    - Very short fragments
    - Words that don't exist in English
    """
    if not text or len(text.strip()) < 2:
        return False
    
    text = text.strip()
    
    # Count alphanumeric vs total
    alphanumeric = sum(1 for c in text if c.isalnum() or c.isspace())
    total = len(text)
    
    # If less than 50% alphanumeric, likely garbage
    if total > 0 and alphanumeric / total < 0.5:
        return False
    
    # Check for common table data patterns FIRST (these are definitely valid)
    table_patterns = [
        r'\$[\d,]+\.?\d*',           # Currency
        r'\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}',  # Dates
        r'\(\d{3}\)\s*\d{3}[-.]?\d{4}',  # Phone numbers
        r'\d+%',                     # Percentages
        r'\d{3,}',                   # 3+ digit numbers
    ]
    
    for pattern in table_patterns:
        if re.search(pattern, text):
            return True  # Definitely valid data
    
    # 🔧 Check for OCR garbage patterns
    def looks_like_garbage(word):
        """Check if a single word looks like OCR garbage."""
        if len(word) <= 3:
            return False  # Short words are OK
        
        word_upper = word.upper()
        word_lower = word.lower()
        
        # Check for repeated unusual patterns
        unusual_doubles = ['II', 'UU', 'AA', 'IIU', 'UUI', 'IUI']
        for pattern in unusual_doubles:
            if pattern in word_upper and word_upper not in ['HAWAII', 'SKIING', 'RADII']:
                return True
        
        # Count consonants in a row
        max_consonants = 0
        current_consonants = 0
        for c in word_lower:
            if c.isalpha() and c not in 'aeiou':
                current_consonants += 1
                max_consonants = max(max_consonants, current_consonants)
            else:
                current_consonants = 0
        
        # 4+ consonants in a row is suspicious for English
        if max_consonants >= 4:
            if not any(p in word_lower for p in ['str', 'thr', 'sch', 'chr', 'spr', 'ngth']):
                return True
        
        # Check for no vowels in a long word
        if len(word) > 4 and not any(c in word_lower for c in 'aeiou'):
            return True
        
        # Check for all-caps words that look like misread text
        if word.isupper() and len(word) > 5:
            vowel_count = sum(1 for c in word_lower if c in 'aeiou')
            vowel_ratio = vowel_count / len(word) if len(word) > 0 else 0
            if vowel_ratio < 0.2 or vowel_ratio > 0.6:
                return True
        
        return False
    
    # Check each word
    words = text.split()
    garbage_words = 0
    valid_words = 0
    
    for word in words:
        clean_word = ''.join(c for c in word if c.isalpha())
        if len(clean_word) < 2:
            continue
        if looks_like_garbage(clean_word):
            garbage_words += 1
        else:
            valid_words += 1
    
    # If ANY garbage words detected, be more careful
    total_words = garbage_words + valid_words
    if total_words > 0:
        garbage_ratio = garbage_words / total_words
        if garbage_ratio >= 0.5:
            return False
        if garbage_words >= 1 and total_words <= 3:
            return False
    
    # Check for names (capitalized words with vowels)
    if re.search(r'[A-Z][a-z]+', text):
        return True
    
    # Check for short abbreviations (2-4 uppercase chars)
    if re.search(r'\b[A-Z]{2,4}\b', text):
        return True
    
    # Check for any numbers
    if re.search(r'\d+', text):
        return True
    
    # Check for recognizable words with vowels
    for word in words:
        clean_word = ''.join(c for c in word if c.isalpha())
        if len(clean_word) >= 3:
            if any(c in clean_word.lower() for c in 'aeiou'):
                if not looks_like_garbage(clean_word):
                    return True
    
    return False


def is_footer_disclaimer(text: str) -> bool:
    """Detect if text is a footer disclaimer (should NOT be in tables)."""
    if not text:
        return False
    
    text_lower = text.lower()
    
    footer_indicators = [
        'please call', 'support center', 'confidential', 'privileged',
        'attorney', 'if you received this', 'intended recipient',
        'not receive all', 'prohibited', 'thank you', 'fax number',
        'copyright', 'all rights reserved', 'page 1 of', 'page _ of',
    ]
    
    for indicator in footer_indicators:
        if indicator in text_lower:
            return True
    
    return False


def is_form_label_only(text: str) -> bool:
    """Detect if text is just a form field label (like "DATE:", "NAME:")."""
    if not text:
        return True
    
    text = text.strip()
    
    if text.endswith(':'):
        return True
    
    form_labels = [
        'date', 'name', 'from', 'to', 'subject', 're', 'cc',
        'phone', 'fax', 'email', 'address', 'city', 'state', 'zip',
        'total', 'amount', 'quantity', 'price', 'description',
        'year', 'month', 'day', 'time', 'signature',
    ]
    
    text_lower = text.lower().rstrip(':').strip()
    if text_lower in form_labels:
        return True
    
    return False


def calculate_table_content_score(cells_text: list) -> float:
    """Calculate a quality score for table content (0.0 to 1.0)."""
    if not cells_text:
        return 0.0
    
    valid_cells = 0
    footer_cells = 0
    label_only_cells = 0
    
    for text in cells_text:
        if not text or len(text.strip()) < 2:
            continue
        
        if is_footer_disclaimer(text):
            footer_cells += 1
        elif is_form_label_only(text):
            label_only_cells += 1
        elif is_likely_table_content(text):
            valid_cells += 1
    
    total = len(cells_text)
    if total == 0:
        return 0.0
    
    if footer_cells > 0:
        return 0.1
    
    if label_only_cells > valid_cells:
        return 0.2
    
    return valid_cells / total


# =============================================================================
# ORIGINAL FUNCTIONS (with enhancements)
# =============================================================================

def find_lines(binarized_image_inv, min_len=400, gap=10, threshold=200):
    """
    ENHANCED: Finds horizontal and vertical lines in the image using Hough Transform.
    Now applies morphological operations to connect broken/faint lines first.

    CRITICAL: min_len=400 and threshold=200 prevent handwriting strokes from being
    detected as table lines. Real tables have long lines (>400px); handwriting does not.
    """
    # NEW: Apply morphological operations to connect broken lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    enhanced_horizontal = cv2.dilate(binarized_image_inv, horizontal_kernel, iterations=1)
    
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    enhanced_vertical = cv2.dilate(binarized_image_inv, vertical_kernel, iterations=1)
    
    enhanced_image = cv2.bitwise_or(enhanced_horizontal, enhanced_vertical)
    
    edges = cv2.Canny(255 - enhanced_image, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=threshold, minLineLength=min_len, maxLineGap=gap
    )
    
    horizontal, vertical = [], []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            if abs(y2 - y1) <= 3:
                horizontal.append((x1, y1, x2, y2))
            elif abs(x2 - x1) <= 3:
                vertical.append((x1, y1, x2, y2))
                
    return horizontal, vertical


def merge_lines(lines, axis='h', tol=5):
    """Merges lines that are aligned and close to each other."""
    if not lines:
        return []

    if axis == 'h':
        sorted_lines = sorted(lines, key=lambda l: ((l[1] + l[3]) / 2, l[0]))
        groups = []
        current_group = [sorted_lines[0]]
        
        for line in sorted_lines[1:]:
            prev_y = (current_group[-1][1] + current_group[-1][3]) / 2
            curr_y = (line[1] + line[3]) / 2
            if abs(curr_y - prev_y) <= tol:
                current_group.append(line)
            else:
                groups.append(current_group)
                current_group = [line]
        groups.append(current_group)
        
        merged = []
        for group in groups:
            y_coords = [(l[1] + l[3]) / 2 for l in group]
            x1 = min(min(l[0], l[2]) for l in group)
            x2 = max(max(l[0], l[2]) for l in group)
            y = int(np.median(y_coords))
            merged.append((x1, y, x2, y))
        return merged
    else:
        sorted_lines = sorted(lines, key=lambda l: ((l[0] + l[2]) / 2, l[1]))
        groups = []
        current_group = [sorted_lines[0]]

        for line in sorted_lines[1:]:
            prev_x = (current_group[-1][0] + current_group[-1][2]) / 2
            curr_x = (line[0] + line[2]) / 2
            if abs(curr_x - prev_x) <= tol:
                current_group.append(line)
            else:
                groups.append(current_group)
                current_group = [line]
        groups.append(current_group)

        merged = []
        for group in groups:
            x_coords = [(l[0] + l[2]) / 2 for l in group]
            y1 = min(min(l[1], l[3]) for l in group)
            y2 = max(max(l[1], l[3]) for l in group)
            x = int(np.median(x_coords))
            merged.append((x, y1, x, y2))
        return merged


def lines_to_cells(horizontal_lines, vertical_lines, min_cell_wh=14):
    """Creates cell bounding boxes from merged horizontal and vertical lines."""
    if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
        return []

    y_coords = sorted({y for _, y, _, _ in horizontal_lines})
    x_coords = sorted({x for x, _, _, _ in vertical_lines})
    
    cells = []
    for r in range(len(y_coords) - 1):
        for c in range(len(x_coords) - 1):
            x1, y1 = x_coords[c], y_coords[r]
            x2, y2 = x_coords[c+1], y_coords[r+1]
            if (x2 - x1) >= min_cell_wh and (y2 - y1) >= min_cell_wh:
                cells.append([float(x1), float(y1), float(x2), float(y2)])
    return cells


def validate_table_structure(cells, text_blocks, page_width, page_height, occupied_cell_ratio_threshold=0.2, iou_threshold=0.8):
    """
    Validates whether a grid is a real table or a form based on the distribution
    of text blocks within it. Includes strict checks to prevent false positives.
    """
    if not cells or not text_blocks:
        return False

    total_cells = len(cells)

    if total_cells < 4:
        return False

    if cells:
        x_coords = [cell[0] for cell in cells] + [cell[2] for cell in cells]
        y_coords = [cell[1] for cell in cells] + [cell[3] for cell in cells]
        table_x1, table_y1 = min(x_coords), min(y_coords)
        table_x2, table_y2 = max(x_coords), max(y_coords)
        table_area = (table_x2 - table_x1) * (table_y2 - table_y1)
        page_area = page_width * page_height

        coverage_ratio = table_area / page_area if page_area > 0 else 0

        if coverage_ratio > 0.65:
            x_centers = [(cell[0] + cell[2]) / 2 for cell in cells]
            x_centers_sorted = sorted(set(x_centers))

            distinct_cols = 1
            if len(x_centers_sorted) > 1:
                prev_x = x_centers_sorted[0]
                for x in x_centers_sorted[1:]:
                    if abs(x - prev_x) > 30:
                        distinct_cols += 1
                    prev_x = x

            if distinct_cols < 3:
                return False

    y_centers = [(cell[1] + cell[3]) / 2 for cell in cells]
    x_centers = [(cell[0] + cell[2]) / 2 for cell in cells]

    y_sorted = sorted(set(y_centers))
    distinct_rows = 1
    if len(y_sorted) > 1:
        prev_y = y_sorted[0]
        for y in y_sorted[1:]:
            if abs(y - prev_y) > 20:
                distinct_rows += 1
            prev_y = y

    x_sorted = sorted(set(x_centers))
    distinct_cols = 1
    if len(x_sorted) > 1:
        prev_x = x_sorted[0]
        for x in x_sorted[1:]:
            if abs(x - prev_x) > 30:
                distinct_cols += 1
            prev_x = x

    if distinct_cols < 2:
        return False

    occupied_cells = set()

    for tb in text_blocks:
        tb_box = tb['bounding_box']
        tb_area = (tb_box[2] - tb_box[0]) * (tb_box[3] - tb_box[1])
        if tb_area == 0: continue

        for i, cell_box in enumerate(cells):
            intersection_area = max(0, min(tb_box[2], cell_box[2]) - max(tb_box[0], cell_box[0])) * \
                                max(0, min(tb_box[3], cell_box[3]) - max(tb_box[1], cell_box[1]))

            if (intersection_area / tb_area) > iou_threshold:
                occupied_cells.add(i)
                break

    occupied_ratio = len(occupied_cells) / total_cells

    if occupied_ratio < occupied_cell_ratio_threshold:
        return False

    return True


def index_cells(cells, axis_tol=10):
    """Indexes cell list as (row, col) and returns a grid map."""
    if not cells:
        return {}, 0, 0

    centers_x = [utils.bbox_center(c['bbox'])[0] for c in cells]
    centers_y = [utils.bbox_center(c['bbox'])[1] for c in cells]
    
    col_centers = utils.cluster_1d(centers_x, tol=axis_tol)
    row_centers = utils.cluster_1d(centers_y, tol=axis_tol)
    
    def find_nearest_idx(val, centers):
        arr = np.array(centers) if centers else np.array([0])
        return int(np.argmin(np.abs(arr - val)))

    grid_map = {}
    for cell in cells:
        cx, cy = utils.bbox_center(cell['bbox'])
        r_idx = find_nearest_idx(cy, row_centers)
        c_idx = find_nearest_idx(cx, col_centers)
        
        cell['row'] = r_idx
        cell['col'] = c_idx
        
        grid_map.setdefault((r_idx, c_idx), []).append(cell)

    num_rows = len(row_centers)
    num_cols = len(col_centers)
        
    return grid_map, num_rows, num_cols


def grid_to_dataframe(grid_map, num_rows, num_cols):
    """Converts grid map to a pandas DataFrame."""
    if not grid_map:
        return pd.DataFrame()

    data = [["" for _ in range(num_cols)] for _ in range(num_rows)]
    
    for (r, c), cell_list in grid_map.items():
        full_text = " | ".join(
            utils.normalize_text_basic(cell.get("text", "")) for cell in cell_list
        )
        if 0 <= r < num_rows and 0 <= c < num_cols:
            data[r][c] = full_text
            
    return pd.DataFrame(data)


# =============================================================================
# 🔧 IMPROVED: BORDERLESS TABLE DETECTION
# =============================================================================

def find_borderless_table_blocks(text_blocks, min_cols=2, min_rows=2, page_width=None, page_height=None):
    """
    🔧 IMPROVED: Finds borderless tables with better validation.
    
    Key improvements:
    1. Excludes header/footer regions
    2. Validates content quality
    3. Prioritizes data-rich regions
    4. Filters out garbage OCR
    """
    if len(text_blocks) < min_cols * min_rows:
        return []

    # 🔧 FIX 1: Filter out header/footer regions
    exclude_top_percent = 0.08
    exclude_bottom_percent = 0.15
    
    if page_height:
        top_cutoff = page_height * exclude_top_percent
        bottom_cutoff = page_height * (1 - exclude_bottom_percent)
        
        filtered_blocks = []
        for block in text_blocks:
            bbox = block.get('bounding_box', [0, 0, 0, 0])
            block_center_y = (bbox[1] + bbox[3]) / 2
            
            # Skip blocks in header/footer regions
            if block_center_y < top_cutoff or block_center_y > bottom_cutoff:
                continue
            
            # 🔧 FIX 2: Skip blocks with footer disclaimer content
            content = block.get('content', '')
            if is_footer_disclaimer(content):
                continue
            
            filtered_blocks.append(block)
        
        text_blocks = filtered_blocks
    
    if len(text_blocks) < min_cols * min_rows:
        return []

    text_blocks = sorted(text_blocks, key=lambda b: (b['bounding_box'][1], b['bounding_box'][0]))
    
    # Group into rows
    rows = []
    if not text_blocks:
        return []
        
    current_row = [text_blocks[0]]
    for block in text_blocks[1:]:
        prev_block = current_row[-1]
        y_center_prev = utils.bbox_center(prev_block['bounding_box'])[1]
        y_center_curr = utils.bbox_center(block['bounding_box'])[1]
        
        if abs(y_center_curr - y_center_prev) < 20:
            current_row.append(block)
        else:
            rows.append(sorted(current_row, key=lambda b: b['bounding_box'][0]))
            current_row = [block]
    rows.append(sorted(current_row, key=lambda b: b['bounding_box'][0]))

    # Find best table candidate with content scoring
    best_table_blocks = []
    best_score = 0

    for i in range(len(rows)):
        num_cols = len(rows[i])
        if num_cols < min_cols:
            continue
        
        current_table_blocks = list(rows[i])
        last_row_in_table = rows[i]

        for j in range(i + 1, len(rows)):
            if len(rows[j]) >= min_cols:
                aligned_cols_count = 0
                for k in range(min(len(last_row_in_table), len(rows[j]))):
                    x_center1 = utils.bbox_center(last_row_in_table[k]['bounding_box'])[0]
                    x_center2 = utils.bbox_center(rows[j][k]['bounding_box'])[0]
                    if abs(x_center1 - x_center2) < 50:
                        aligned_cols_count += 1

                if aligned_cols_count / min(len(last_row_in_table), len(rows[j])) > 0.5:
                    current_table_blocks.extend(rows[j])
                    last_row_in_table = rows[j]
                else:
                    break
            else:
                break
        
        # 🔧 FIX 3: Calculate content quality score
        cells_text = [b.get('content', '') for b in current_table_blocks]
        content_score = calculate_table_content_score(cells_text)
        
        # Skip tables with poor content quality
        if content_score < 0.3:
            continue
        
        num_rows_in_candidate = len(set(tuple(b['bounding_box']) for b in current_table_blocks)) / num_cols if num_cols > 0 else 0
        if len(current_table_blocks) > best_score and num_rows_in_candidate >= min_rows:
            # Weight by content quality
            overall_score = len(current_table_blocks) * content_score
            if overall_score > best_score:
                best_score = overall_score
                best_table_blocks = current_table_blocks

    # 🔧 FIX 4: Final validation - check table doesn't span too much of page
    if best_table_blocks and page_width and page_height:
        all_x = [coord for block in best_table_blocks for coord in [block['bounding_box'][0], block['bounding_box'][2]]]
        all_y = [coord for block in best_table_blocks for coord in [block['bounding_box'][1], block['bounding_box'][3]]]

        table_x1, table_y1 = min(all_x), min(all_y)
        table_x2, table_y2 = max(all_x), max(all_y)
        table_area = (table_x2 - table_x1) * (table_y2 - table_y1)
        page_area = page_width * page_height

        coverage_ratio = table_area / page_area if page_area > 0 else 0

        x_centers = [utils.bbox_center(block['bounding_box'])[0] for block in best_table_blocks]
        x_centers_sorted = sorted(set(x_centers))

        distinct_cols = 1
        if len(x_centers_sorted) > 1:
            prev_x = x_centers_sorted[0]
            for x in x_centers_sorted[1:]:
                if abs(x - prev_x) > 30:
                    distinct_cols += 1
                prev_x = x

        # Reject if table covers > 65% of page and has fewer than 3 columns
        if coverage_ratio > 0.65 and distinct_cols < 3:
            return []
        
        # 🔧 FIX 5: Also reject if > 70% coverage regardless of columns
        if coverage_ratio > 0.70:
            return []

    return best_table_blocks


def detect_borderless_table_advanced(text_blocks, page_width, page_height):
    """
    Advanced borderless table detection using DBSCAN clustering.
    """
    if len(text_blocks) < 4:
        return None

    try:
        from sklearn.cluster import DBSCAN
    except ImportError:
        return None

    # 🔧 FIX: Filter header/footer before clustering
    exclude_top = page_height * 0.08
    exclude_bottom = page_height * 0.85
    
    filtered_blocks = [
        b for b in text_blocks 
        if exclude_top < (b['bounding_box'][1] + b['bounding_box'][3]) / 2 < exclude_bottom
        and not is_footer_disclaimer(b.get('content', ''))
    ]
    
    if len(filtered_blocks) < 4:
        return None

    boxes = [block['bounding_box'] for block in filtered_blocks]
    centers_x = [(b[0] + b[2]) / 2 for b in boxes]
    centers_y = [(b[1] + b[3]) / 2 for b in boxes]

    x_array = np.array(centers_x).reshape(-1, 1)
    col_clustering = DBSCAN(eps=30, min_samples=2).fit(x_array)
    col_labels = col_clustering.labels_
    n_columns = len(set(col_labels)) - (1 if -1 in col_labels else 0)

    if n_columns < 2:
        return None

    y_array = np.array(centers_y).reshape(-1, 1)
    row_clustering = DBSCAN(eps=25, min_samples=2).fit(y_array)
    row_labels = row_clustering.labels_
    n_rows = len(set(row_labels)) - (1 if -1 in row_labels else 0)

    if n_rows < 2:
        return None

    grid = {}
    for i, block in enumerate(filtered_blocks):
        if col_labels[i] == -1 or row_labels[i] == -1:
            continue
        cell_key = (row_labels[i], col_labels[i])
        if cell_key not in grid:
            grid[cell_key] = []
        grid[cell_key].append(block)

    expected_cells = n_rows * n_columns
    occupied_cells = len(grid)
    occupancy_ratio = occupied_cells / expected_cells if expected_cells > 0 else 0

    if occupancy_ratio < 0.4:
        return None

    # 🔧 FIX: Validate content quality
    all_text = [b.get('content', '') for blocks in grid.values() for b in blocks]
    content_score = calculate_table_content_score(all_text)
    if content_score < 0.3:
        return None

    return {
        'type': 'borderless_table',
        'n_rows': n_rows,
        'n_columns': n_columns,
        'grid': grid,
        'occupancy_ratio': occupancy_ratio,
        'content_score': content_score
    }