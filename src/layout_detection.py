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
    print("Warning: layoutparser not available. Falling back to Tesseract-only layout detection.")

def _initialize_layout_model():
    """Initializes the layout detection model only when first called."""
    global _layout_model
    if _layout_model is None and LAYOUTPARSER_AVAILABLE:
        try:
            print("    - Initializing LayoutParser model (PaddleDetection)...")
            # Using PaddleDetection model for layout analysis
            _layout_model = lp.models.PaddleDetectionLayoutModel(
                config_path="lp://PubLayNet/ppyolov2_r50vd_dcn_365e_publaynet/config",
                threshold=0.5,
                label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
                enforce_cpu=False,  # Use GPU if available
                enable_mkldnn=True
            )
            print("    - ✅ LayoutParser model initialized successfully.")
        except Exception as e:
            print(f"    - [WARNING] Failed to initialize LayoutParser model: {e}")
            print("    - Falling back to Tesseract-only layout detection.")
            _layout_model = None

def group_words_into_blocks(df, tolerance=20):
    """
    Groups word DataFrame from Tesseract into semantic text blocks.
    """
    if df.empty:
        return []

    df['line_group'] = (df['top'].diff() > tolerance).cumsum()

    blocks = []
    for _, group in df.groupby('line_group'):
        x1 = group['left'].min()
        y1 = group['top'].min()
        x2 = (group['left'] + group['width']).max()
        y2 = (group['top'] + group['height']).max()
        
        text_lines = []
        for block_num, line_df in group.groupby(['block_num', 'line_num']):
            line_text = ' '.join(line_df['text'].astype(str))
            text_lines.append(line_text)
        
        full_text = '\n'.join(text_lines)

        blocks.append({
            "bounding_box": [float(x1), float(y1), float(x2), float(y2)],
            "content": utils.normalize_text_basic(full_text),
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
    NEW: Uses LayoutParser with a pre-trained model to detect layout elements.
    Returns a list of layout elements with their types and bounding boxes.
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
        
        # Sort by reading order (top to bottom, left to right)
        elements.sort(key=lambda e: (e['bounding_box'][1], e['bounding_box'][0]))
        
        return elements
        
    except Exception as e:
        print(f"    - Error during layout detection: {e}")
        return None

def detect_text_blocks(bgr_image, lang=DEFAULT_OCR_LANG):
    """
    ENHANCED: First tries LayoutParser model, falls back to Tesseract if unavailable.
    """
    # Try using layoutparser model first
    layout_elements = detect_layout_with_model(bgr_image)
    
    if layout_elements is not None:
        print("    - Using LayoutParser model for layout analysis")
        return layout_elements
    else:
        print("    - Using Tesseract for layout analysis")
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