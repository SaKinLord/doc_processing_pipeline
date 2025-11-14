# src/ocr.py

import re
import cv2
import torch
import pytesseract
from .config import DEFAULT_OCR_LANG, LOW_CONFIDENCE_THRESHOLD

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


def _initialize_paddle():
    """Initializes PaddleOCR only when first called. Checks ModelManager first."""
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
    if _paddle_ocr_instance is None and 'paddleocr' in globals():
        print("    - Initializing PaddleOCR for the first time...")
        _paddle_ocr_instance = PaddleOCR(use_angle_cls=True, lang='latin', show_log=False)
        print("    - PaddleOCR initialized.")

def _initialize_trocr():
    """Initializes TrOCR model and processor only when first called. Checks ModelManager first."""
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
    if _trocr_processor_instance is None and 'TrOCRProcessor' in globals() and torch.cuda.is_available():
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


def _clean_text(text):
    """Cleans excess whitespace and OCR-related noise from text."""
    if not isinstance(text, str):
        return ""
    return re.sub(r'\s+', ' ', text).strip()

def _get_psm_for_box(box):
    """Selects the most appropriate Tesseract PSM mode based on the box's aspect ratio."""
    x1, y1, x2, y2 = map(float, box)
    w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
    aspect_ratio = w / h
    if aspect_ratio > 5.0: return 7
    if w * h < 8000: return 13
    return 6

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
        print("Error: Tesseract is not installed or not in your PATH.")
        return "", 0.0
    if data is None or data.empty: return "", 0.0
    data.dropna(subset=['text'], inplace=True)
    data = data[data.conf != -1]
    text = " ".join([str(t) for t in data['text'].tolist() if str(t).strip()])
    confidence = float(data['conf'].mean()) if not data.empty else 0.0
    return _clean_text(text), confidence

def ocr_with_paddle(roi_bgr):
    """Runs PaddleOCR on the given ROI."""
    _initialize_paddle()
    if _paddle_ocr_instance is None: return ""
    rgb_image = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    try:
        result = _paddle_ocr_instance.ocr(rgb_image, cls=True)
        if not result or not result[0]: return ""
        texts = [line[1][0] for line in result[0] if line and len(line) > 1]
        return _clean_text(" ".join(texts))
    except Exception as e:
        print(f"    - Error during PaddleOCR: {e}")
        return ""

def ocr_with_trocr(roi_bgr):
    """Runs TrOCR model on the given ROI."""
    _initialize_trocr()
    if _trocr_model_instance is None or _trocr_processor_instance is None:
        return ""
    
    rgb_image = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    try:
        pixel_values = _trocr_processor_instance(images=rgb_image, return_tensors="pt").pixel_values.to("cuda")
        generated_ids = _trocr_model_instance.generate(pixel_values)
        generated_text = _trocr_processor_instance.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return _clean_text(generated_text)
    except Exception as e:
        print(f"    - Error during TrOCR inference: {e}")
        return ""

def ocr_smart(original_bgr_image, binary_image, box, lang=DEFAULT_OCR_LANG):
    """
    Applies 3-stage smart OCR: Tesseract -> PaddleOCR -> TrOCR.
    Each stage activates when the previous one fails.
    """
    x1, y1, x2, y2 = map(int, box)
    roi_binary = binary_image[y1:y2, x1:x2]
    if roi_binary.size == 0: return "", 0.0

    # --- Stage 1: Tesseract (optimized for printed text) ---
    tess_text, tess_conf = ocr_with_tesseract(roi_binary, lang=lang, box=box)
    
    # If Tesseract result is good enough, no need to continue.
    if tess_conf > 85.0 and len(tess_text) > 3:
        return tess_text, tess_conf

    # Variables to store the result
    best_text = tess_text
    best_conf = tess_conf

    roi_original = original_bgr_image[y1:y2, x1:x2]
    if roi_original.size == 0: return best_text, best_conf

    # --- Stage 2: PaddleOCR Fallback (general purpose and handwriting) ---
    if (tess_conf < LOW_CONFIDENCE_THRESHOLD or len(tess_text) < 5) and 'paddleocr' in globals():
        print(f"    - Tesseract low ({tess_conf:.2f}%). Trying PaddleOCR...")
        paddle_text = ocr_with_paddle(roi_original)
        
        if len(paddle_text) > len(best_text) + 3:
            print(f"    - PaddleOCR provided a better result. Using it for now.")
            best_text = paddle_text
            best_conf = max(tess_conf, 75.0)

    # --- Stage 3: TrOCR Fallback (handwriting expert) ---
    if (best_conf < LOW_CONFIDENCE_THRESHOLD or len(best_text) < 5) and 'TrOCRProcessor' in globals():
        print(f"    - Previous results still weak. Triggering TrOCR expert...")
        trocr_text = ocr_with_trocr(roi_original)

        if len(trocr_text) > len(best_text) + 3:
            print(f"    - ✅ TrOCR provided the best result!")
            best_text = trocr_text
            best_conf = 90.0
            
    return best_text, best_conf