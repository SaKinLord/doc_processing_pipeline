# src/table_extraction.py

import cv2
import numpy as np
import pandas as pd
from . import utils

def find_lines(binarized_image_inv, min_len=80, gap=10, threshold=120):
    """
    ENHANCED: Finds horizontal and vertical lines in the image using Hough Transform.
    Now applies morphological operations to connect broken/faint lines first.
    """
    # NEW: Apply morphological operations to connect broken lines
    # This makes faint, dashed, or broken lines more solid and easier to detect
    
    # Connect horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    enhanced_horizontal = cv2.dilate(binarized_image_inv, horizontal_kernel, iterations=1)
    
    # Connect vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    enhanced_vertical = cv2.dilate(binarized_image_inv, vertical_kernel, iterations=1)
    
    # Combine both enhancements
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
    else: # axis == 'v'
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

def validate_table_structure(cells, text_blocks, occupied_cell_ratio_threshold=0.2, iou_threshold=0.8):
    """
    Validates whether a grid is a real table or a form based on the distribution
    of text blocks within it.
    """
    if not cells or not text_blocks:
        return False

    total_cells = len(cells)
    occupied_cells = set()

    for tb in text_blocks:
        tb_box = tb['bounding_box']
        tb_area = (tb_box[2] - tb_box[0]) * (tb_box[3] - tb_box[1])
        if tb_area == 0: continue

        for i, cell_box in enumerate(cells):
            # Check how much of the text block is inside the cell
            intersection_area = max(0, min(tb_box[2], cell_box[2]) - max(tb_box[0], cell_box[0])) * \
                                max(0, min(tb_box[3], cell_box[3]) - max(tb_box[1], cell_box[1]))
            
            if (intersection_area / tb_area) > iou_threshold:
                occupied_cells.add(i)
                break # A text block belongs to only one cell

    occupied_ratio = len(occupied_cells) / total_cells
    
    # If the occupancy rate of cells is too low, this is probably a form.
    if occupied_ratio < occupied_cell_ratio_threshold:
        return False
        
    return True

def index_cells(cells, axis_tol=10):
    """
    Indexes cell list as (row, col) and returns a grid map.
    """
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

def find_borderless_table_blocks(text_blocks, min_cols=2, min_rows=2):
    """
    Finds a group of blocks forming a table based on text block alignment.
    """
    if len(text_blocks) < min_cols * min_rows:
        return []

    text_blocks.sort(key=lambda b: (b['bounding_box'][1], b['bounding_box'][0]))
    
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

    max_cells = 0
    best_table_blocks = []

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
        
        num_rows_in_candidate = len(set(tuple(b['bounding_box']) for b in current_table_blocks)) / num_cols if num_cols > 0 else 0
        if len(current_table_blocks) > max_cells and num_rows_in_candidate >= min_rows:
            max_cells = len(current_table_blocks)
            best_table_blocks = current_table_blocks
            
    return best_table_blocks