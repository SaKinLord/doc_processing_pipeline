# src/layout_detection.py
"""
Layout Detection Module

🆕 ARCHITECTURE CHANGE (v2.1):
Surya OCR is now the PRIMARY layout detector for finding text bounding boxes.
Tesseract is only used as a fallback when Surya is unavailable.

Pipeline:
1. Surya Detection → Find all text line bounding boxes (superior line detection)
2. Tesseract/TrOCR → Read text from Surya's detected regions (recognition only)

This fixes the issue where Tesseract missed lines in noisy fax documents.
"""

import cv2
import numpy as np
import pandas as pd
import pytesseract
from typing import List, Dict, Tuple, Optional

try:
    from . import utils
    from .config import DEFAULT_OCR_LANG, USE_LAYOUTPARSER, USE_SURYA_OCR
except ImportError:
    import utils
    from config import DEFAULT_OCR_LANG, USE_LAYOUTPARSER, USE_SURYA_OCR

# =============================================================================
#  🆕 SURYA-BASED LAYOUT DETECTION (PRIMARY)
# =============================================================================

# Surya model instances (lazy loading)
_surya_det_model = None
_surya_det_processor = None
SURYA_DETECTION_AVAILABLE = False
SURYA_API_VERSION = None

# Surya function references
_surya_batch_text_detection = None
_surya_load_model = None
_surya_load_processor = None

def _check_surya_availability():
    """
    Check if Surya OCR is available.
    
    Surya 0.6.0 API:
    - surya.detection.batch_text_detection(images, model, processor)
    - surya.model.detection.model.load_model()
    - surya.model.detection.model.load_processor()  # NOTE: In model module, not processor!
    """
    global SURYA_DETECTION_AVAILABLE, SURYA_API_VERSION
    global _surya_batch_text_detection, _surya_load_model, _surya_load_processor
    
    try:
        # Import the detection function
        from surya.detection import batch_text_detection
        _surya_batch_text_detection = batch_text_detection
        print("    - ✅ Found surya.detection.batch_text_detection")
        
        # Import model and processor loaders
        # IMPORTANT: In surya 0.6.0, load_processor is in the MODEL module!
        from surya.model.detection.model import load_model, load_processor
        _surya_load_model = load_model
        _surya_load_processor = load_processor
        print("    - ✅ Found load_model and load_processor in surya.model.detection.model")
        
        SURYA_DETECTION_AVAILABLE = True
        SURYA_API_VERSION = 'v0.6'
        print("    - ✅ Surya OCR detection available (v0.6.0 API)")
        return True
        
    except ImportError as e:
        print(f"    - ❌ Surya import error: {e}")
        SURYA_DETECTION_AVAILABLE = False
        return False
    except Exception as e:
        print(f"    - ❌ Surya error: {type(e).__name__}: {e}")
        SURYA_DETECTION_AVAILABLE = False
        return False

# Run the check at module load time
_check_surya_availability()


def _initialize_surya_detection():
    """Initialize Surya detection model (lazy loading)."""
    global _surya_det_model, _surya_det_processor
    
    if not SURYA_DETECTION_AVAILABLE:
        return False
    
    if _surya_det_model is not None:
        return True
    
    try:
        print("    - Initializing Surya Detection model...")
        
        # Load model and processor using the discovered functions
        _surya_det_model = _surya_load_model()
        _surya_det_processor = _surya_load_processor()
        
        print("    - ✅ Surya Detection model initialized successfully")
        return True
        
    except Exception as e:
        print(f"    - ⚠️ Failed to initialize Surya Detection: {e}")
        import traceback
        traceback.print_exc()
        return False


def detect_text_blocks_surya(bgr_image: np.ndarray) -> List[Dict]:
    """
    🆕 PRIMARY LAYOUT DETECTION using Surya.
    
    Surya excels at finding text lines in:
    - Noisy fax documents
    - Multi-column layouts
    - Dense text regions
    - Documents with mixed orientations
    
    Args:
        bgr_image: BGR image (numpy array)
    
    Returns:
        List of text block dictionaries with bounding boxes
    """
    if not SURYA_DETECTION_AVAILABLE or not USE_SURYA_OCR:
        print("    - Surya not available, falling back to Tesseract layout detection.")
        return None
    
    if not _initialize_surya_detection():
        return None
    
    try:
        from PIL import Image
        
        # Convert BGR to RGB PIL Image
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        # Run Surya text detection
        # API: batch_text_detection(images, model, processor)
        predictions = _surya_batch_text_detection(
            [pil_image],
            _surya_det_model,
            _surya_det_processor
        )
        
        if not predictions or len(predictions) == 0:
            print("    - Surya detected no text regions.")
            return []
        
        result = predictions[0]
        blocks = []
        
        # Extract bounding boxes from Surya's TextDetectionResult
        # The result has .bboxes attribute containing TextLine objects
        detection_items = []
        if hasattr(result, 'bboxes'):
            detection_items = result.bboxes
        elif hasattr(result, 'text_lines'):
            detection_items = result.text_lines
        elif isinstance(result, list):
            detection_items = result
        
        for text_line in detection_items:
            bbox = None
            confidence = 0.9
            
            # Extract bbox from TextLine object
            if hasattr(text_line, 'bbox'):
                bbox = text_line.bbox
            elif hasattr(text_line, 'polygon'):
                # Convert polygon to bbox
                poly = text_line.polygon
                if poly and len(poly) >= 4:
                    xs = [p[0] for p in poly]
                    ys = [p[1] for p in poly]
                    bbox = [min(xs), min(ys), max(xs), max(ys)]
            elif isinstance(text_line, (list, tuple)) and len(text_line) >= 4:
                bbox = text_line[:4]
            
            if hasattr(text_line, 'confidence'):
                confidence = text_line.confidence
            
            if bbox and len(bbox) >= 4:
                x1, y1, x2, y2 = map(float, bbox[:4])
                
                # Filter out tiny detections (noise)
                width = x2 - x1
                height = y2 - y1
                if width < 10 or height < 5:
                    continue
                
                blocks.append({
                    'bounding_box': [x1, y1, x2, y2],
                    'type': 'TextBlock',
                    'content': '',
                    'confidence': confidence,
                    'detection_source': 'surya'
                })
        
        print(f"    - ✅ Surya detected {len(blocks)} text regions.")
        return blocks
        
    except Exception as e:
        print(f"    - ⚠️ Surya detection error: {e}")
        import traceback
        traceback.print_exc()
        return None


def detect_text_lines_surya(bgr_image: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Get raw text line bounding boxes from Surya.
    
    Returns:
        List of (x1, y1, x2, y2) tuples for each detected text line
    """
    blocks = detect_text_blocks_surya(bgr_image)
    if blocks is None:
        return []
    
    return [tuple(map(int, b['bounding_box'])) for b in blocks]


# =============================================================================
#  TESSERACT-BASED LAYOUT DETECTION (FALLBACK)
# =============================================================================

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
        if hasattr(utils, 'normalize_text_basic'):
            full_text = utils.normalize_text_basic(full_text)

        blocks.append({
            "bounding_box": [float(x1), float(y1), float(x2), float(y2)],
            "content": full_text,
            "type": "TextBlock",
            "detection_source": "tesseract"
        })
    return blocks


def detect_text_blocks_tesseract(bgr_image, lang=DEFAULT_OCR_LANG):
    """
    FALLBACK: Uses Tesseract's PSM 3 mode to analyze page layout.
    Only used when Surya is unavailable.
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


# =============================================================================
#  🆕 UNIFIED LAYOUT DETECTION (Surya-first with Tesseract fallback)
# =============================================================================

def detect_text_blocks(bgr_image: np.ndarray, lang: str = DEFAULT_OCR_LANG) -> List[Dict]:
    """
    🆕 UNIFIED LAYOUT DETECTION - Surya-first architecture.
    
    Pipeline:
    1. Try Surya detection first (superior line finding)
    2. Fall back to Tesseract only if Surya fails
    
    Args:
        bgr_image: BGR image (numpy array)
        lang: Language code for Tesseract fallback
    
    Returns:
        List of text block dictionaries with bounding boxes
    """
    # Step 1: Try Surya (PRIMARY)
    if USE_SURYA_OCR and SURYA_DETECTION_AVAILABLE:
        blocks = detect_text_blocks_surya(bgr_image)
        if blocks is not None and len(blocks) > 0:
            return blocks
        elif blocks is not None and len(blocks) == 0:
            # Surya ran but found nothing - might be blank page
            print("    - Surya found no text, trying Tesseract as backup...")
    
    # Step 2: Tesseract fallback
    print("    - Using Tesseract for layout detection (fallback).")
    return detect_text_blocks_tesseract(bgr_image, lang)


def detect_layout_elements(bgr_image: np.ndarray, lang: str = DEFAULT_OCR_LANG) -> Dict:
    """
    Detect all layout elements including text blocks, tables, and figures.
    
    Returns:
        Dictionary with 'text_blocks', 'table_regions', 'figure_regions'
    """
    result = {
        'text_blocks': [],
        'table_regions': [],
        'figure_regions': [],
        'detection_method': 'surya' if USE_SURYA_OCR else 'tesseract'
    }
    
    # Get text blocks using Surya-first detection
    result['text_blocks'] = detect_text_blocks(bgr_image, lang)
    
    # LayoutParser for tables/figures (if enabled)
    if USE_LAYOUTPARSER:
        layout_elements = detect_layout_with_model(bgr_image)
        if layout_elements:
            for elem in layout_elements:
                elem_type = elem.get('type', '')
                if elem_type == 'Table':
                    result['table_regions'].append(elem)
                elif elem_type == 'Figure':
                    result['figure_regions'].append(elem)
    
    return result


# =============================================================================
#  LAYOUTPARSER INTEGRATION (for Tables/Figures only)
# =============================================================================

# LayoutParser model instance
_layout_model = None

try:
    import layoutparser as lp
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


def detect_layout_with_model(bgr_image):
    """
    Uses LayoutParser with a pre-trained model to detect layout elements.
    Primarily for detecting tables and figures (NOT text blocks).
    """
    _initialize_layout_model()

    if _layout_model is None:
        return None
    
    try:
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        
        # Run layout detection
        layout_result = _layout_model.detect(rgb_image)
        
        elements = []
        for block in layout_result:
            elem_type = block.type
            
            # Map LayoutParser types to our types
            if elem_type in ['Table', 'table']:
                our_type = 'Table'
            elif elem_type in ['Figure', 'figure', 'Image', 'image']:
                our_type = 'Figure'
            else:
                continue  # Skip text blocks (Surya handles those)
            
            x1, y1, x2, y2 = block.block.x_1, block.block.y_1, block.block.x_2, block.block.y_2
            
            elements.append({
                'type': our_type,
                'bounding_box': [float(x1), float(y1), float(x2), float(y2)],
                'confidence': float(block.score) if hasattr(block, 'score') else 0.5,
                'layout_type': elem_type,
                'detection_source': 'layoutparser'
            })
        
        return _apply_nms(elements) if elements else None
        
    except Exception as e:
        print(f"    - LayoutParser detection failed: {e}")
        return None


def _calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
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


def _apply_nms(elements, iou_threshold=0.5):
    """Apply Non-Maximum Suppression to remove overlapping detections."""
    if not elements:
        return elements

    by_class = {}
    for elem in elements:
        class_type = elem.get('layout_type', elem['type'])
        if class_type not in by_class:
            by_class[class_type] = []
        by_class[class_type].append(elem)

    all_kept = []
    for class_type, class_elements in by_class.items():
        sorted_elements = sorted(class_elements, key=lambda x: x['confidence'], reverse=True)

        keep = []
        while sorted_elements:
            current = sorted_elements.pop(0)
            keep.append(current)

            filtered = []
            for elem in sorted_elements:
                iou = _calculate_iou(current['bounding_box'], elem['bounding_box'])
                if iou < iou_threshold:
                    filtered.append(elem)

            sorted_elements = filtered

        all_kept.extend(keep)

    return all_kept


# =============================================================================
#  READING ORDER FUNCTIONS
# =============================================================================

def reorder_blocks_by_columns(blocks, tolerance=80):
    """
    Reorders blocks by column-aware reading order.
    """
    if not blocks:
        return blocks
    
    blocks_with_center = []
    for b in blocks:
        bbox = b['bounding_box']
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        blocks_with_center.append((b, cx, cy))
    
    # Sort by y first, then by x within similar y
    blocks_with_center.sort(key=lambda x: (x[2] // tolerance, x[1]))
    
    return [b[0] for b in blocks_with_center]


def reorder_blocks_smart(blocks, use_graph=True):
    """
    Smart reading order using graph-based topological sorting.
    """
    if not blocks or len(blocks) <= 1:
        return blocks
    
    if not use_graph:
        return reorder_blocks_by_columns(blocks)
    
    # Simple column-based approach for now
    # TODO: Implement full graph-based ordering
    return reorder_blocks_by_columns(blocks)


# =============================================================================
#  UTILITY FUNCTIONS
# =============================================================================

def merge_overlapping_blocks(blocks: List[Dict], iou_threshold: float = 0.3) -> List[Dict]:
    """
    Merge text blocks that significantly overlap.
    Useful when Surya detects overlapping lines.
    """
    if not blocks or len(blocks) <= 1:
        return blocks
    
    merged = []
    used = set()
    
    for i, block1 in enumerate(blocks):
        if i in used:
            continue
        
        current_box = list(block1['bounding_box'])
        merged_content = [block1.get('content', '')]
        used.add(i)
        
        for j, block2 in enumerate(blocks):
            if j in used or j == i:
                continue
            
            iou = _calculate_iou(current_box, block2['bounding_box'])
            if iou > iou_threshold:
                # Merge boxes
                b2 = block2['bounding_box']
                current_box[0] = min(current_box[0], b2[0])
                current_box[1] = min(current_box[1], b2[1])
                current_box[2] = max(current_box[2], b2[2])
                current_box[3] = max(current_box[3], b2[3])
                merged_content.append(block2.get('content', ''))
                used.add(j)
        
        merged.append({
            'bounding_box': current_box,
            'type': 'TextBlock',
            'content': ' '.join(filter(None, merged_content)),
            'detection_source': block1.get('detection_source', 'unknown')
        })
    
    return merged


def filter_blocks_by_size(blocks: List[Dict], min_width: int = 10, min_height: int = 5) -> List[Dict]:
    """
    Filter out blocks that are too small (likely noise).
    """
    filtered = []
    for block in blocks:
        bbox = block['bounding_box']
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        if width >= min_width and height >= min_height:
            filtered.append(block)
    return filtered
