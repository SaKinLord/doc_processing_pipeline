# src/layout_detection.py

import cv2
import numpy as np
import pandas as pd
import pytesseract
from . import utils
from .config import DEFAULT_OCR_LANG

# NEW: LayoutParser integration
try:
    import layoutparser as lp
    _layout_model = None
    LAYOUTPARSER_AVAILABLE = True
except ImportError:
    LAYOUTPARSER_AVAILABLE = False

def _initialize_layout_model():
    """Initializes the layout detection model via ModelManager."""
    global _layout_model
    if _layout_model is None and LAYOUTPARSER_AVAILABLE:
        try:
            from .model_manager import ModelManager
            _layout_model = ModelManager.get_layout_model()
        except Exception:
            _layout_model = None

def group_words_into_blocks(df, tolerance=20):
    """
    Groups word DataFrame from Tesseract into semantic text blocks.
    Respects columns by checking horizontal gaps.
    """
    if df.empty:
        return []

    # Sort by top then left to handle reading order roughly
    df = df.sort_values(by=['top', 'left'])

    # Calculate vertical difference between consecutive words
    df['v_diff'] = df['top'].diff().fillna(0)
    
    # Calculate horizontal difference (gap) between consecutive words
    df['prev_right'] = (df['left'] + df['width']).shift(1).fillna(0)
    df['h_diff'] = df['left'] - df['prev_right']
    
    # Define thresholds
    V_THRESHOLD = tolerance
    H_THRESHOLD = 100  # Large gap indicating column break
    
    is_new_block = (df['v_diff'] > V_THRESHOLD) | \
                   ((df['v_diff'].abs() < 20) & (df['h_diff'] > H_THRESHOLD))
                   
    df['line_group'] = is_new_block.cumsum()

    blocks = []
    for _, group in df.groupby('line_group'):
        x1 = group['left'].min()
        y1 = group['top'].min()
        x2 = (group['left'] + group['width']).max()
        y2 = (group['top'] + group['height']).max()
        
        full_text = " ".join(group['text'].astype(str))
        full_text = utils.normalize_text_basic(full_text)

        blocks.append({
            "bounding_box": [float(x1), float(y1), float(x2), float(y2)],
            "content": full_text,
            "type": "TextBlock"
        })
    return blocks

def detect_text_blocks_tesseract(bgr_image, lang=DEFAULT_OCR_LANG):
    """
    Uses Tesseract's PSM 3 mode to analyze page layout and
    returns text blocks, bounding boxes, and contents.
    """
    try:
        data = pytesseract.image_to_data(
            bgr_image, lang=lang, config="--oem 3 --psm 3",
            output_type=pytesseract.Output.DATAFRAME
        )
    except pytesseract.TesseractNotFoundError:
        print("Error: Tesseract is not installed or not in your PATH.")
        return []

    if data is None or data.empty:
        return []

    data.dropna(subset=['text'], inplace=True)
    data = data[data.conf > 30]
    data = data[data.text.str.strip().astype(bool)]

    if data.empty:
        return []

    blocks = group_words_into_blocks(data, tolerance=20)
    blocks.sort(key=lambda b: (b['bounding_box'][1], b['bounding_box'][0]))
    
    return blocks

def detect_layout_with_model(bgr_image):
    """
    Uses LayoutParser with a pre-trained model to detect layout elements.
    """
    _initialize_layout_model()
    
    if _layout_model is None:
        return None
    
    try:
        # Convert BGR to RGB for layoutparser
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        
        # Detect layout
        layout = _layout_model.detect(rgb_image)
        
        elements = []
        for block in layout:
            x1, y1, x2, y2 = block.coordinates
            element_type = block.type
            
            # Map layoutparser types to our types
            type_map = {
                "Text": "TextBlock",
                "Title": "TextBlock",
                "List": "TextBlock",
                "Table": "Table",
                "Figure": "Figure"
            }
            
            elements.append({
                "bounding_box": [float(x1), float(y1), float(x2), float(y2)],
                "type": type_map.get(element_type, "TextBlock"),
                "layout_type": element_type,
                "confidence": float(block.score)
            })
        
        # Sort by reading order
        elements.sort(key=lambda e: (e['bounding_box'][1], e['bounding_box'][0]))
        
        return elements
        
    except Exception as e:
        print(f"    - [WARNING] LayoutParser failed for this page: {e}")
        return None

def detect_text_blocks(bgr_image, lang=DEFAULT_OCR_LANG):
    """
    First tries LayoutParser model, falls back to Tesseract if unavailable.
    """
    # Try using layoutparser model first
    layout_elements = detect_layout_with_model(bgr_image)
    
    if layout_elements is not None:
        # print("    - Using LayoutParser model for layout analysis")
        return layout_elements
    else:
        print("    - Using Tesseract for layout analysis (Fallback)")
        return detect_text_blocks_tesseract(bgr_image, lang)

def reorder_blocks_by_columns(blocks, col_tolerance=80):
    if not blocks:
        return []

    centers_x = [utils.bbox_center(b['bounding_box'])[0] for b in blocks]
    col_centers = utils.cluster_1d(centers_x, tol=col_tolerance)

    def find_nearest_col_idx(val, centers):
        arr = np.array(centers)
        return int(np.argmin(np.abs(arr - val)))

    for block in blocks:
        cx = utils.bbox_center(block['bounding_box'])[0]
        block['_col_idx'] = find_nearest_col_idx(cx, col_centers)

    sorted_blocks = sorted(blocks, key=lambda b: (b['_col_idx'], b['bounding_box'][1]))

    for block in sorted_blocks:
        del block['_col_idx']

    return sorted_blocks


def reorder_blocks_by_graph(blocks, vertical_threshold=20, horizontal_overlap_ratio=0.3):
    """
    Reorders text blocks using graph-based topological sorting.
    """
    if not blocks or len(blocks) <= 1:
        return blocks

    n = len(blocks)
    graph = {i: [] for i in range(n)}
    in_degree = {i: 0 for i in range(n)}

    def get_horizontal_overlap(box1, box2):
        x1_min, _, x1_max, _ = box1
        x2_min, _, x2_max, _ = box2
        overlap_start = max(x1_min, x2_min)
        overlap_end = min(x1_max, x2_max)
        if overlap_end <= overlap_start: return 0.0
        overlap_width = overlap_end - overlap_start
        min_width = min(x1_max - x1_min, x2_max - x2_min)
        if min_width <= 0: return 0.0
        return overlap_width / min_width

    def get_vertical_overlap(box1, box2):
        _, y1_min, _, y1_max = box1
        _, y2_min, _, y2_max = box2
        overlap_start = max(y1_min, y2_min)
        overlap_end = min(y1_max, y2_max)
        if overlap_end <= overlap_start: return 0.0
        overlap_height = overlap_end - overlap_start
        min_height = min(y1_max - y1_min, y2_max - y2_min)
        if min_height <= 0: return 0.0
        return overlap_height / min_height

    for i in range(n):
        box_i = blocks[i]['bounding_box']
        _, y1_i, _, y2_i = box_i
        center_y_i = (y1_i + y2_i) / 2

        for j in range(n):
            if i == j: continue
            box_j = blocks[j]['bounding_box']
            _, y1_j, _, y2_j = box_j
            
            h_overlap = get_horizontal_overlap(box_i, box_j)
            if h_overlap > horizontal_overlap_ratio:
                if y2_i < y1_j - vertical_threshold:
                    graph[i].append(j)
                    in_degree[j] += 1

            v_overlap = get_vertical_overlap(box_i, box_j)
            if v_overlap > 0.5:
                x1_i = box_i[0]
                x1_j = box_j[0]
                if x1_i < x1_j and h_overlap < 0.1:
                    graph[i].append(j)
                    in_degree[j] += 1

    queue = []
    for i in range(n):
        if in_degree[i] == 0: queue.append(i)

    queue.sort(key=lambda idx: (blocks[idx]['bounding_box'][1], blocks[idx]['bounding_box'][0]))
    sorted_indices = []
    
    while queue:
        queue.sort(key=lambda idx: (blocks[idx]['bounding_box'][1], blocks[idx]['bounding_box'][0]))
        current = queue.pop(0)
        sorted_indices.append(current)

        for neighbor in graph[current]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(sorted_indices) != n:
        remaining = set(range(n)) - set(sorted_indices)
        remaining_sorted = sorted(remaining, key=lambda idx: (blocks[idx]['bounding_box'][1], blocks[idx]['bounding_box'][0]))
        sorted_indices.extend(remaining_sorted)

    return [blocks[i] for i in sorted_indices]


def reorder_blocks_smart(blocks, use_graph=True, col_tolerance=80):
    if not blocks: return []
    if len(blocks) <= 2:
        return sorted(blocks, key=lambda b: (b['bounding_box'][1], b['bounding_box'][0]))

    if use_graph:
        try:
            return reorder_blocks_by_graph(blocks)
        except Exception as e:
            print(f"    - Graph-based reordering failed: {e}. Falling back to column-based.")
            return reorder_blocks_by_columns(blocks, col_tolerance)
    else:
        return reorder_blocks_by_columns(blocks, col_tolerance)