# src/ocr.py

import re
import cv2
import torch
import pytesseract
import numpy as np

try:
    from .config import (
        DEFAULT_OCR_LANG, LOW_CONFIDENCE_THRESHOLD, ENABLE_TEXT_POSTPROCESSING,
        USE_SURYA_OCR, SURYA_COMPLEXITY_THRESHOLD, SURYA_MIN_LINE_FALLBACK,
        SURYA_LINE_DETECTION_ONLY, SURYA_CONFIDENCE_THRESHOLD,
        USE_FUZZY_MATCHING
    )
    from . import image_utils
    from . import text_postprocessing
except ImportError:
    from config import (
        DEFAULT_OCR_LANG, LOW_CONFIDENCE_THRESHOLD, ENABLE_TEXT_POSTPROCESSING,
        USE_SURYA_OCR, SURYA_COMPLEXITY_THRESHOLD, SURYA_MIN_LINE_FALLBACK,
        SURYA_LINE_DETECTION_ONLY, SURYA_CONFIDENCE_THRESHOLD,
        USE_FUZZY_MATCHING
    )
    import image_utils
    import text_postprocessing

# --- PaddleOCR Integration ---
try:
    from paddleocr import PaddleOCR
    _paddle_ocr_instance = None
except ImportError:
    _paddle_ocr_instance = None
    print("Warning: paddleocr not found. Handwriting recognition (Paddle fallback) will be disabled.")

# --- TrOCR Integration ---
try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    _trocr_processor_instance = None
    _trocr_model_instance = None
except ImportError:
    _trocr_processor_instance = None
    _trocr_model_instance = None
    print("Warning: sentencepiece or other transformers dependencies not found. TrOCR fallback will be disabled.")

# --- Handwriting Detector Integration ---
try:
    from .handwriting_detector import get_handwriting_detector
    _handwriting_detector_available = True
except ImportError:
    _handwriting_detector_available = False
    print("Warning: handwriting_detector module not available. Smart OCR routing will use heuristics only.")

# =============================================================================
#  🆕 SURYA OCR INTEGRATION (Feature 2)
# =============================================================================
_surya_predictor = None
_surya_det_processor = None
_surya_rec_model = None
_surya_det_model = None
SURYA_AVAILABLE = False
SURYA_OCR_API_VERSION = None

# Surya function references
_surya_batch_text_detection = None
_surya_load_det_model = None
_surya_load_det_processor = None
_surya_load_rec_model = None
_surya_load_rec_processor = None

def _check_surya_ocr_availability():
    """
    Check if Surya OCR is available.
    
    Surya 0.6.0 API:
    - surya.detection.batch_text_detection(images, model, processor)
    - surya.model.detection.model.load_model()
    - surya.model.detection.model.load_processor()  # NOTE: In model module!
    - surya.model.recognition.model.load_model()
    - surya.model.recognition.processor.load_processor()
    """
    global SURYA_AVAILABLE, SURYA_OCR_API_VERSION
    global _surya_batch_text_detection
    global _surya_load_det_model, _surya_load_det_processor
    global _surya_load_rec_model, _surya_load_rec_processor
    
    try:
        # Import detection function
        from surya.detection import batch_text_detection
        _surya_batch_text_detection = batch_text_detection
        print("    - ✅ Found surya.detection.batch_text_detection")
        
        # Import detection model and processor loaders
        # NOTE: In surya 0.6.0, load_processor is in the MODEL module!
        from surya.model.detection.model import load_model as load_det_model
        from surya.model.detection.model import load_processor as load_det_processor
        _surya_load_det_model = load_det_model
        _surya_load_det_processor = load_det_processor
        print("    - ✅ Found detection load_model and load_processor")
        
        # Import recognition model and processor loaders (optional)
        try:
            from surya.model.recognition.model import load_model as load_rec_model
            from surya.model.recognition.processor import load_processor as load_rec_processor
            _surya_load_rec_model = load_rec_model
            _surya_load_rec_processor = load_rec_processor
            print("    - ✅ Found recognition load_model and load_processor")
        except ImportError:
            print("    - ⚠️ Recognition models not available (detection-only mode)")
        
        SURYA_AVAILABLE = True
        SURYA_OCR_API_VERSION = 'v0.6'
        print("    - ✅ Surya OCR available (v0.6.0 API)")
        return True
        
    except ImportError as e:
        print(f"    - ❌ Surya import error: {e}")
        SURYA_AVAILABLE = False
        return False
    except Exception as e:
        print(f"    - ❌ Surya error: {type(e).__name__}: {e}")
        SURYA_AVAILABLE = False
        return False

# Run availability check at module load
_check_surya_ocr_availability()


def _initialize_surya():
    """Initialize Surya OCR models (lazy loading)."""
    global _surya_det_model, _surya_det_processor, _surya_rec_model, _surya_predictor
    
    if not SURYA_AVAILABLE or not USE_SURYA_OCR:
        return False
    
    if _surya_det_model is not None:
        return True
    
    try:
        print("    - Initializing Surya OCR models...")
        
        # Load detection model and processor
        _surya_det_model = _surya_load_det_model()
        _surya_det_processor = _surya_load_det_processor()
        print("    - ✅ Detection model loaded")
        
        # Load recognition model if available and not line-detection-only
        if not SURYA_LINE_DETECTION_ONLY and _surya_load_rec_model is not None:
            try:
                _surya_rec_model = _surya_load_rec_model()
                _surya_predictor = _surya_load_rec_processor()
                print("    - ✅ Recognition model loaded")
            except Exception as e:
                print(f"    - ⚠️ Recognition model not loaded: {e}")
        
        print("    - ✅ Surya OCR initialized successfully")
        return True
        
    except Exception as e:
        print(f"    - ⚠️ Failed to initialize Surya OCR: {e}")
        import traceback
        traceback.print_exc()
        return False


def ocr_with_surya(image_bgr, langs=None):
    """
    Run Surya OCR on the given image.
    
    Args:
        image_bgr: BGR image (numpy array)
        langs: List of language codes (default: ['en'])
    
    Returns:
        tuple: (text, confidence, line_bboxes)
    """
    if not SURYA_AVAILABLE or not USE_SURYA_OCR:
        return "", 0.0, []
    
    if not _initialize_surya():
        return "", 0.0, []
    
    if langs is None:
        langs = ['en']
    
    try:
        from PIL import Image
        rgb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        # Run detection: batch_text_detection(images, model, processor)
        predictions = _surya_batch_text_detection(
            [pil_image],
            _surya_det_model,
            _surya_det_processor
        )
        
        if not predictions or len(predictions) == 0:
            return "", 0.0, []
        
        # Extract results
        result = predictions[0]
        text_lines = []
        confidences = []
        line_bboxes = []
        
        for text_line in result.text_lines:
            if hasattr(text_line, 'text') and text_line.text:
                text_lines.append(text_line.text)
                
                if hasattr(text_line, 'confidence'):
                    confidences.append(text_line.confidence)
                else:
                    confidences.append(0.85)  # Default confidence
                
                if hasattr(text_line, 'bbox'):
                    line_bboxes.append(text_line.bbox)
        
        full_text = " ".join(text_lines)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return full_text, avg_confidence * 100, line_bboxes
        
    except Exception as e:
        print(f"    - Surya OCR error: {e}")
        return "", 0.0, []


def get_surya_line_detections(image_bgr):
    """
    Use Surya for line detection only.
    
    Args:
        image_bgr: BGR image (numpy array)
    
    Returns:
        list: List of line bounding boxes [(x1, y1, x2, y2), ...]
    """
    if not SURYA_AVAILABLE or not USE_SURYA_OCR:
        return []
    
    if not _initialize_surya():
        return []
    
    try:
        from PIL import Image
        
        rgb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        # Run detection: batch_text_detection(images, model, processor)
        predictions = _surya_batch_text_detection(
            [pil_image],
            _surya_det_model,
            _surya_det_processor
        )
        
        if not predictions or len(predictions) == 0:
            return []
        
        result = predictions[0]
        line_bboxes = []
        
        # Extract bboxes from result
        detection_items = []
        if hasattr(result, 'bboxes'):
            detection_items = result.bboxes
        elif hasattr(result, 'text_lines'):
            detection_items = result.text_lines
        elif isinstance(result, list):
            detection_items = result
        
        for text_line in detection_items:
            bbox = None
            if hasattr(text_line, 'bbox'):
                bbox = text_line.bbox
            elif hasattr(text_line, 'polygon'):
                poly = text_line.polygon
                if poly and len(poly) >= 4:
                    xs = [p[0] for p in poly]
                    ys = [p[1] for p in poly]
                    bbox = [min(xs), min(ys), max(xs), max(ys)]
            elif isinstance(text_line, (list, tuple)) and len(text_line) >= 4:
                bbox = text_line[:4]
            
            if bbox and len(bbox) >= 4:
                line_bboxes.append([int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])])
        
        return line_bboxes
        
    except Exception as e:
        print(f"    - Surya line detection error: {e}")
        import traceback
        traceback.print_exc()
        return []


# Helper functions for extracting Surya results (kept for compatibility)
def _extract_surya_items(result):
    """Extract detection items from Surya result."""
    for attr in ['bboxes', 'text_lines', 'lines', 'detections', 'boxes']:
        if hasattr(result, attr):
            return getattr(result, attr)
    if isinstance(result, dict):
        for key in ['bboxes', 'text_lines', 'lines', 'detections', 'boxes']:
            if key in result:
                return result[key]
    if isinstance(result, list):
        return result
    return []


def _extract_surya_bbox(text_line):
    """Extract bounding box from Surya detection item."""
    if hasattr(text_line, 'bbox'):
        return text_line.bbox
    if hasattr(text_line, 'bounding_box'):
        return text_line.bounding_box
    if isinstance(text_line, dict):
        return text_line.get('bbox') or text_line.get('bounding_box')
    if isinstance(text_line, (list, tuple)) and len(text_line) >= 4:
        try:
            [float(x) for x in text_line[:4]]
            return text_line[:4]
        except:
            pass
    if hasattr(text_line, 'polygon'):
        poly = text_line.polygon
        if poly and len(poly) >= 4:
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            return [min(xs), min(ys), max(xs), max(ys)]
    return None


def is_complex_layout(quality_metrics, text_blocks_count=0):
    """
    Determine if the document has a complex layout that benefits from Surya.
    
    Args:
        quality_metrics: Image quality analysis dictionary
        text_blocks_count: Number of text blocks detected by primary OCR
    
    Returns:
        bool: True if layout is complex
    """
    # High text density suggests complex form or dense document
    text_density = quality_metrics.get('text_density', 0.1)
    if text_density > SURYA_COMPLEXITY_THRESHOLD:
        return True
    
    # Very few lines detected might indicate missed content
    if text_blocks_count < SURYA_MIN_LINE_FALLBACK:
        return True
    
    # Low local contrast uniformity suggests mixed regions
    local_uniformity = quality_metrics.get('local_contrast_uniformity', 30)
    if local_uniformity > 40:  # High variance = complex layout
        return True
    
    return False


# =============================================================================
#  EXISTING OCR FUNCTIONS (with Surya integration points)
# =============================================================================

def _initialize_paddle():
    """Initializes PaddleOCR only when first called."""
    global _paddle_ocr_instance
    if _paddle_ocr_instance is not None:
        return
    
    # Try to get from ModelManager first
    try:
        from .model_manager import ModelManager
        _paddle_ocr_instance = ModelManager.get_paddle_ocr()
        if _paddle_ocr_instance is not None:
            return
    except:
        pass
    
    # Fall back to on-demand initialization
    if _paddle_ocr_instance is None and 'PaddleOCR' in dir():
        print("    - Initializing PaddleOCR for the first time...")
        _paddle_ocr_instance = PaddleOCR(use_angle_cls=True, lang='latin', show_log=False)
        print("    - PaddleOCR initialized.")


def _initialize_trocr():
    """Initializes TrOCR model and processor only when first called."""
    global _trocr_processor_instance, _trocr_model_instance
    if _trocr_processor_instance is not None:
        return
    
    # Try to get from ModelManager first
    try:
        from .model_manager import ModelManager
        _trocr_processor_instance, _trocr_model_instance = ModelManager.get_trocr()
        if _trocr_processor_instance is not None:
            return
    except:
        pass
    
    # Fall back to on-demand initialization
    if _trocr_processor_instance is None and 'TrOCRProcessor' in dir() and torch.cuda.is_available():
        try:
            print("    - Initializing TrOCR model for handwriting (this may take a moment)...")
            model_id = "microsoft/trocr-base-handwritten"
            _trocr_processor_instance = TrOCRProcessor.from_pretrained(model_id)
            _trocr_model_instance = VisionEncoderDecoderModel.from_pretrained(model_id).to("cuda")
            print("    - ✅ TrOCR model initialized successfully on GPU.")
        except Exception as e:
            print(f"    - [ERROR] Failed to initialize TrOCR model: {e}")
            _trocr_processor_instance = None
            _trocr_model_instance = None


# =============================================================================
# OCR RETRY WITH PREPROCESSING STRATEGIES
# =============================================================================

def enhance_contrast(image):
    """CLAHE-based contrast enhancement for low-contrast images."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def sharpen_image(image):
    """Unsharp masking for sharpening blurry images."""
    gaussian = cv2.GaussianBlur(image, (0, 0), 3)
    return cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)


def morphology_clean(image):
    """Clean noise with morphological operations."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    return cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)


def scale_image_2x(image):
    """2x upscaling with cubic interpolation for small text."""
    return cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)


def ocr_with_retry(original_bgr, binary_image, box, lang, min_confidence=50.0):
    """Attempts OCR with multiple preprocessing strategies if confidence is low."""
    strategies = [
        ("original", lambda img: img),
        ("contrast_enhanced", enhance_contrast),
        ("sharpened", sharpen_image),
        ("scaled_2x", scale_image_2x),
        ("morphology_cleaned", morphology_clean),
    ]

    best_result = ("", 0.0)

    for strategy_name, preprocess_fn in strategies:
        try:
            processed = preprocess_fn(original_bgr)

            if len(processed.shape) == 3:
                gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            else:
                gray = processed

            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 45, 15
            )

            text, conf = ocr_with_tesseract(processed, binary, box, lang)

            if conf > best_result[1]:
                best_result = (text, conf)

            if conf >= min_confidence:
                return text, conf

        except Exception:
            continue

    return best_result


def _clean_text(text):
    """Cleans excess whitespace and OCR-related noise from text."""
    if not isinstance(text, str):
        return ""
    return re.sub(r'\s+', ' ', text).strip()


def split_text_block_into_lines(image_bgr):
    """Splits a text block into separate lines using horizontal projection profiles."""
    if image_bgr is None or image_bgr.size == 0:
        return []

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    h_projection = np.sum(binary, axis=1)
    threshold = np.max(h_projection) * 0.1
    
    in_line = False
    line_start = 0
    line_regions = []
    
    for i, val in enumerate(h_projection):
        if not in_line and val > threshold:
            in_line = True
            line_start = i
        elif in_line and val <= threshold:
            in_line = False
            if i - line_start > 5:
                line_regions.append((line_start, i))
    
    if in_line:
        line_regions.append((line_start, len(h_projection)))
    
    line_images = []
    for start, end in line_regions:
        padding = 3
        start = max(0, start - padding)
        end = min(image_bgr.shape[0], end + padding)
        line_img = image_bgr[start:end, :]
        if line_img.shape[0] > 5 and line_img.shape[1] > 5:
            line_images.append(line_img)
    
    return line_images


def _calculate_text_quality_score(text, confidence):
    """Calculates a quality score based on text characteristics and OCR confidence."""
    if not text or len(text.strip()) == 0:
        return 0.0

    score = 0.0
    score += confidence * 0.4

    words = text.split()
    if len(words) > 0:
        single_char_words = sum(1 for w in words if len(w) == 1)
        word_ratio = 1.0 - (single_char_words / len(words))
        score += word_ratio * 20.0

    valid_words = sum(1 for w in words if len(w) >= 2 and any(c.isalpha() for c in w))
    if len(words) > 0:
        score += (valid_words / len(words)) * 20.0

    if len(text) > 10:
        special_chars = sum(1 for c in text if not c.isalnum() and c not in ' .,;:!?-\'\"')
        special_ratio = special_chars / len(text)
        score += max(0, (1.0 - special_ratio * 5)) * 10.0

    if len(text) > 0:
        alnum_count = sum(1 for c in text if c.isalnum() or c.isspace())
        alnum_ratio = alnum_count / len(text)
        score += alnum_ratio * 10.0

    return min(100.0, score)


def _is_likely_handwritten(text, confidence):
    """Heuristic to determine if text is likely handwritten based on OCR results."""
    if not text or len(text.strip()) == 0:
        return False

    words = text.split()
    if len(words) == 0:
        return False

    avg_word_length = len(text.replace(' ', '')) / len(words)

    if avg_word_length < 2.5 and confidence < 70.0:
        return True

    single_chars = sum(1 for w in words if len(w) == 1)
    if len(words) > 2 and single_chars / len(words) > 0.35:
        return True
    
    if re.search(r'\b[li1I]\s+[li1I]\s+[li1I]\b', text):
        return True

    return False


def _get_psm_for_box(box):
    """Selects the most appropriate Tesseract PSM mode."""
    x1, y1, x2, y2 = map(float, box)
    w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
    aspect_ratio = w / h
    if aspect_ratio > 5.0:
        return 7
    if w * h < 8000:
        return 13
    return 6


def _detect_text_type(roi_image):
    """Uses the handwriting detector to classify text as printed or handwritten."""
    if not _handwriting_detector_available:
        return 'unknown', 0.0

    try:
        detector = get_handwriting_detector()
        result = detector.classify(roi_image)
        return result['classification'], result['confidence']
    except Exception as e:
        print(f"    - Handwriting detection failed: {e}")
        return 'unknown', 0.0


def ocr_routed(original_bgr_image, binary_image, box, lang=DEFAULT_OCR_LANG, quality_metrics=None, sr_applied=False):
    """
    Routes OCR to the most appropriate engine based on text type detection.
    
    🆕 Enhanced with Surya OCR for complex layouts.
    🆕 PERFORMANCE FIX: Added sr_applied flag to skip redundant SR on handwriting regions.
    """
    x1, y1, x2, y2 = map(int, box)
    
    h, w = binary_image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    if x2 <= x1 or y2 <= y1:
        return "", 0.0

    roi_original = original_bgr_image[y1:y2, x1:x2]
    roi_binary = binary_image[y1:y2, x1:x2]

    if roi_original.size == 0:
        return "", 0.0

    # Step 1: Detect text type
    text_type, detection_confidence = _detect_text_type(roi_original)

    # Step 2: Route to appropriate OCR engine
    if text_type == 'handwritten' and detection_confidence > 0.50:
        print(f"      - Region classified as handwritten (conf: {detection_confidence:.2f})")
        
        # 🆕 Pass sr_applied flag to skip redundant SR
        handwriting_ready_img = image_utils.preprocess_for_handwriting(
            roi_original, 
            quality_metrics=None,  # Don't pass metrics if SR already applied
            full_image_sr_applied=sr_applied
        )

        if _trocr_processor_instance is not None or 'TrOCRProcessor' in dir():
            trocr_text = ocr_with_trocr(handwriting_ready_img)
            if trocr_text and len(trocr_text.strip()) > 0:
                return trocr_text, 85.0

        if _paddle_ocr_instance is not None or 'PaddleOCR' in dir():
            paddle_text = ocr_with_paddle(handwriting_ready_img)
            if paddle_text and len(paddle_text.strip()) > 0:
                return paddle_text, 75.0

        return ocr_with_tesseract(roi_binary, lang=lang, box=box)

    elif text_type == 'printed' and detection_confidence > 0.6:
        # 🆕 Try Surya first for complex layouts
        if USE_SURYA_OCR and quality_metrics and is_complex_layout(quality_metrics):
            print("    - Complex layout detected. Trying Surya OCR...")
            surya_text, surya_conf, _ = ocr_with_surya(roi_original)
            if surya_text and surya_conf > SURYA_CONFIDENCE_THRESHOLD * 100:
                return surya_text, surya_conf
        
        tess_text, tess_conf = ocr_with_tesseract(roi_binary, lang=lang, box=box)
        quality = _calculate_text_quality_score(tess_text, tess_conf)

        if quality > 60.0:
            return tess_text, tess_conf

        if quality < 50.0 and (_paddle_ocr_instance is not None or 'PaddleOCR' in dir()):
            paddle_text = ocr_with_paddle(roi_original)
            if paddle_text:
                paddle_quality = _calculate_text_quality_score(paddle_text, 75.0)
                if paddle_quality > quality:
                    return paddle_text, 75.0

        return tess_text, tess_conf

    else:
        return ocr_smart(original_bgr_image, binary_image, box, lang, quality_metrics, sr_applied)


def ocr_with_tesseract(roi_bgr, lang=DEFAULT_OCR_LANG, psm=None, box=None):
    """Runs Tesseract OCR on the given ROI."""
    if psm is None and box is not None:
        psm = _get_psm_for_box(box)
    elif psm is None:
        psm = 6
    config = f"--oem 3 --psm {psm}"
    try:
        data = pytesseract.image_to_data(roi_bgr, lang=lang, config=config, output_type=pytesseract.Output.DATAFRAME)
    except pytesseract.TesseractNotFoundError:
        return "", 0.0
    if data is None or data.empty:
        return "", 0.0
    data.dropna(subset=['text'], inplace=True)
    data = data[data.conf != -1]
    text = " ".join([str(t) for t in data['text'].tolist() if str(t).strip()])
    confidence = float(data['conf'].mean()) if not data.empty else 0.0
    return _clean_text(text), confidence


def ocr_with_paddle(roi_bgr):
    """Runs PaddleOCR on the given ROI."""
    _initialize_paddle()
    if _paddle_ocr_instance is None:
        return ""
    rgb_image = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    try:
        result = _paddle_ocr_instance.ocr(rgb_image, cls=True)
        if not result or not result[0]:
            return ""
        texts = [line[1][0] for line in result[0] if line and len(line) > 1]
        return _clean_text(" ".join(texts))
    except Exception:
        return ""


def ocr_with_trocr(roi_bgr):
    """Runs TrOCR model on the given ROI with LINE SEGMENTATION."""
    _initialize_trocr()
    if _trocr_model_instance is None or _trocr_processor_instance is None:
        return ""
    
    line_images = split_text_block_into_lines(roi_bgr)
    full_text_parts = []
    
    for line_img in line_images:
        if line_img.shape[0] < 5 or line_img.shape[1] < 5:
            continue

        h, w = line_img.shape[:2]
        if h < 32:
            scale = 32 / h
            line_img = cv2.resize(line_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        rgb_image = cv2.cvtColor(line_img, cv2.COLOR_BGR2RGB)
        try:
            pixel_values = _trocr_processor_instance(images=rgb_image, return_tensors="pt").pixel_values.to("cuda")

            generated_ids = _trocr_model_instance.generate(
                pixel_values,
                max_new_tokens=128,
                num_beams=5,
                early_stopping=True,
                no_repeat_ngram_size=3,
                do_sample=False,
                length_penalty=1.0,
                repetition_penalty=1.2
            )
            generated_text = _trocr_processor_instance.batch_decode(generated_ids, skip_special_tokens=True)[0]

            clean_text = generated_text.strip()
            if clean_text and is_valid_ocr_output(clean_text):
                full_text_parts.append(clean_text)

        except Exception as e:
            print(f"    - Error during TrOCR line inference: {e}")
            continue

    return _clean_text(" ".join(full_text_parts))


def is_valid_ocr_output(text):
    """Validates OCR output to filter hallucinations and garbage."""
    if not text or len(text.strip()) == 0:
        return False

    text = text.strip()

    if len(text) > 5:
        for pattern_len in range(2, min(6, len(text) // 2)):
            pattern = text[:pattern_len]
            repetitions = text.count(pattern)
            if repetitions > len(text) / pattern_len * 0.6:
                return False

    alnum_count = sum(1 for c in text if c.isalnum())
    if len(text) > 3 and alnum_count / len(text) < 0.3:
        return False

    garbage_patterns = [
        r'^[.,:;!?]+$',
        r'^[-_=+]+$',
        r'(.)\1{4,}',
        r'^[A-Z]{15,}$',
    ]

    for pattern in garbage_patterns:
        if re.match(pattern, text):
            return False

    return True


def texts_are_similar(text1, text2, threshold=0.7):
    """Check if two texts are similar using character-level similarity."""
    if not text1 or not text2:
        return False

    def get_bigrams(text):
        text = text.lower().replace(' ', '')
        return set(text[i:i+2] for i in range(len(text) - 1))

    b1, b2 = get_bigrams(text1), get_bigrams(text2)
    if not b1 or not b2:
        return text1.lower() == text2.lower()

    intersection = len(b1 & b2)
    union = len(b1 | b2)

    return intersection / union >= threshold if union > 0 else False


def ocr_ensemble(original_bgr, binary_image, box, lang='eng'):
    """Run multiple OCR engines and combine results using weighted voting."""
    results = []

    try:
        tess_text, tess_conf = ocr_with_tesseract(binary_image, lang=lang, box=box)
        if tess_text.strip():
            quality = _calculate_text_quality_score(tess_text, tess_conf)
            results.append({
                'text': tess_text,
                'confidence': tess_conf,
                'quality': quality,
                'engine': 'tesseract',
                'weight': 0.4 if tess_conf > 70 else 0.2
            })
    except Exception:
        pass

    try:
        paddle_text = ocr_with_paddle(original_bgr)
        if paddle_text.strip():
            quality = _calculate_text_quality_score(paddle_text, 75.0)
            results.append({
                'text': paddle_text,
                'confidence': 75.0,
                'quality': quality,
                'engine': 'paddleocr',
                'weight': 0.35
            })
    except Exception:
        pass

    # 🆕 Add Surya to ensemble if available
    if USE_SURYA_OCR and SURYA_AVAILABLE:
        try:
            surya_text, surya_conf, _ = ocr_with_surya(original_bgr)
            if surya_text.strip():
                quality = _calculate_text_quality_score(surya_text, surya_conf)
                results.append({
                    'text': surya_text,
                    'confidence': surya_conf,
                    'quality': quality,
                    'engine': 'surya',
                    'weight': 0.4
                })
        except Exception:
            pass

    if not results:
        return "", 0.0

    if len(results) == 1:
        return results[0]['text'], results[0]['confidence']

    best = max(results, key=lambda r: r['quality'] * r['weight'])
    return best['text'], best['confidence']


def ocr_smart(original_bgr_image, binary_image, box, lang=DEFAULT_OCR_LANG, quality_metrics=None, sr_applied=False):
    """
    Applies multi-stage smart OCR with Surya integration.
    
    Pipeline: Tesseract -> Surya (if complex) -> PaddleOCR -> TrOCR
    
    🆕 PERFORMANCE FIX: Added sr_applied flag to skip redundant SR on handwriting regions.
    """
    x1, y1, x2, y2 = map(int, box)
    
    h, w = binary_image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    roi_binary = binary_image[y1:y2, x1:x2]
    roi_original = original_bgr_image[y1:y2, x1:x2]

    if roi_binary.size == 0:
        return "", 0.0

    # --- Stage 1: Tesseract ---
    tess_text, tess_conf = ocr_with_tesseract(roi_binary, lang=lang, box=box)

    if tess_conf < 50.0:
        retry_text, retry_conf = ocr_with_retry(roi_original, roi_binary, box, lang, min_confidence=50.0)
        if retry_conf > tess_conf:
            tess_text, tess_conf = retry_text, retry_conf

    tess_quality = _calculate_text_quality_score(tess_text, tess_conf)

    if tess_quality > 80.0:
        return _apply_post_processing(tess_text, lang)

    best_text = tess_text
    best_conf = tess_conf
    best_quality = tess_quality

    # --- Stage 1.5: 🆕 Surya for complex layouts ---
    if USE_SURYA_OCR and SURYA_AVAILABLE and best_quality < 60.0:
        surya_text, surya_conf, _ = ocr_with_surya(roi_original)
        if surya_text:
            surya_quality = _calculate_text_quality_score(surya_text, surya_conf)
            if surya_quality > best_quality + 10:
                best_text = surya_text
                best_conf = surya_conf
                best_quality = surya_quality
                print(f"    - Surya improved quality: {tess_quality:.1f} -> {surya_quality:.1f}")

    is_fragmented = _is_likely_handwritten(best_text, best_conf)
    should_try_alternatives = (best_conf < 60.0 or best_quality < 50.0 or is_fragmented)

    # --- Stage 2: PaddleOCR ---
    if should_try_alternatives and (_paddle_ocr_instance is not None or 'PaddleOCR' in dir()):
        paddle_text = ocr_with_paddle(roi_original)
        if paddle_text:
            paddle_quality = _calculate_text_quality_score(paddle_text, 75.0)
            if paddle_quality > (best_quality + 10): 
                best_text = paddle_text
                best_conf = 75.0
                best_quality = paddle_quality

    # --- Stage 3: TrOCR (Handwriting Expert) ---
    is_still_bad = best_quality < 45.0 or _is_likely_handwritten(best_text, best_conf)
    
    if is_still_bad and (_trocr_processor_instance is not None or 'TrOCRProcessor' in dir()):
        # 🆕 Pass sr_applied flag to skip redundant SR
        hw_roi = image_utils.preprocess_for_handwriting(
            roi_original,
            quality_metrics=None,
            full_image_sr_applied=sr_applied
        )
        trocr_text = ocr_with_trocr(hw_roi)
        if trocr_text:
            trocr_quality = _calculate_text_quality_score(trocr_text, 85.0)
            if trocr_quality > best_quality:
                best_text = trocr_text
                best_conf = 85.0
    
    if best_quality < 20.0:
        return "", 0.0

    return _apply_post_processing(best_text, lang), best_conf


def _apply_post_processing(text, lang):
    """Apply text post-processing including fuzzy matching if enabled."""
    if not ENABLE_TEXT_POSTPROCESSING:
        return text
    
    # Language-aware corrections
    if lang in ['eng', 'fra', 'deu', 'spa', 'por', 'ita']:
        text = text_postprocessing.apply_text_corrections(text)
    else:
        text = text_postprocessing.fix_spacing_issues(text)
    
    # 🆕 Apply fuzzy dictionary matching if enabled
    if USE_FUZZY_MATCHING:
        try:
            from . import dictionary_utils
            text = dictionary_utils.apply_fuzzy_corrections(text)
        except ImportError:
            pass
    
    return text
