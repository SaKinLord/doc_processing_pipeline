# src/layout_detection.py

import cv2
import numpy as np
import pandas as pd
import pytesseract
from . import utils
from .config import DEFAULT_OCR_LANG

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
            "content": utils.normalize_text_basic(full_text)
        })
    return blocks

def detect_text_blocks(bgr_image, lang=DEFAULT_OCR_LANG):
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