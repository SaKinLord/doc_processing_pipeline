# src/ocr.py

import re
import cv2
import torch
import pytesseract
import numpy as np
from .config import DEFAULT_OCR_LANG, LOW_CONFIDENCE_THRESHOLD
from . import image_utils  # NEW: Import image_utils for adaptive preprocessing

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
    if _paddle_ocr_instance is None and 'paddleocr' in globals():
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

def split_text_block_into_lines(image_bgr):
    """
    Splits a text block into separate lines using horizontal projection profiles.
    Critical for TrOCR which expects single-line images.
    """
    if image_bgr is None or image_bgr.size == 0:
        return []

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    # Use Otsu to find text pixels (foreground)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Calculate horizontal projection (sum of pixels along rows)
    proj = np.sum(binary, axis=1)
    
    height, width = gray.shape
    lines = []
    start_idx = None
    
    # Dynamic threshold based on mean pixel density to filter noise
    threshold = np.mean(proj) * 0.1 
    
    for i in range(height):
        if proj[i] > threshold and start_idx is None:
            start_idx = i
        elif proj[i] <= threshold and start_idx is not None:
            # Line ended
            # Ignore noise lines smaller than 8px vertical height
            if (i - start_idx) > 8: 
                # Add padding (5px top/bottom)
                y1 = max(0, start_idx - 5)
                y2 = min(height, i + 5)
                # Extract line
                line_crop = image_bgr[y1:y2, :]
                lines.append(line_crop)
            start_idx = None
            
    # If projection didn't separate lines (e.g., single line or very noisy), return original
    if not lines:
        return [image_bgr]
        
    return lines


def _calculate_text_quality_score(text, confidence=0.0):
    """
    Calculates a quality score for OCR text based on multiple heuristics.
    Returns a score from 0-100.
    """
    if not text or not isinstance(text, str):
        return 0.0

    text = text.strip()
    if len(text) == 0:
        return 0.0

    score = 0.0

    # 1. Confidence contribution (0-30 points)
    score += min(30.0, confidence * 0.3)

    # 2. Length contribution (0-15 points)
    length_score = min(15.0, len(text) / 10.0)
    score += length_score

    # 3. Dictionary check (0-20 points)
    common_words = {
        'the', 'and', 'is', 'in', 'to', 'of', 'it', 'for', 'on', 'with', 'as', 'at', 'by',
        've', 'ile', 'bir', 'bu', 'da', 'de', 'için', 'çok', 'ama', 'gibi', 'var', 'yok',
        'total', 'date', 'invoice', 'amount', 'tax', 'tarih', 'toplam', 'fatura', 'kdv'
    }
    words = text.lower().split()
    if any(w in common_words for w in words):
        score += 20.0

    # 4. Fragmentation penalty
    if len(words) > 0:
        avg_word_length = len(text.replace(' ', '')) / len(words)
        if avg_word_length >= 4.0:
            score += 25.0
        elif avg_word_length >= 3.0:
            score += 15.0
        elif avg_word_length >= 2.0:
            score += 5.0

    # 5. Character composition
    if len(text) > 0:
        alnum_count = sum(1 for c in text if c.isalnum() or c.isspace())
        alnum_ratio = alnum_count / len(text)
        score += alnum_ratio * 10.0

    return min(100.0, score)


def _is_likely_handwritten(text, confidence):
    """
    Heuristic to determine if text is likely handwritten based on OCR results.
    """
    if not text or len(text.strip()) == 0:
        return False

    words = text.split()
    if len(words) == 0:
        return False

    avg_word_length = len(text.replace(' ', '')) / len(words)

    # Highly fragmented with low confidence
    if avg_word_length < 2.5 and confidence < 70.0:
        return True

    # Many single-character "words"
    single_chars = sum(1 for w in words if len(w) == 1)
    if len(words) > 2 and single_chars / len(words) > 0.35:
        return True
    
    # Check for specific noise patterns
    if re.search(r'\b[li1I]\s+[li1I]\s+[li1I]\b', text):
        return True

    return False


def _get_psm_for_box(box):
    """Selects the most appropriate Tesseract PSM mode."""
    x1, y1, x2, y2 = map(float, box)
    w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
    aspect_ratio = w / h
    if aspect_ratio > 5.0:
        return 7 # Single line
    if w * h < 8000:
        return 13 # Raw line
    return 6 # Block


def _detect_text_type(roi_image):
    """
    Uses the handwriting detector to classify text as printed or handwritten.
    """
    if not _handwriting_detector_available:
        return 'unknown', 0.0

    try:
        detector = get_handwriting_detector()
        result = detector.classify(roi_image)
        return result['classification'], result['confidence']
    except Exception as e:
        print(f"    - Handwriting detection failed: {e}")
        return 'unknown', 0.0


def ocr_routed(original_bgr_image, binary_image, box, lang=DEFAULT_OCR_LANG):
    """
    Routes OCR to the most appropriate engine based on text type detection.
    
    Includes ADAPTIVE PROCESSING: Uses specific image preprocessing for handwriting.
    """
    x1, y1, x2, y2 = map(int, box)
    
    # Coordinate safety
    h, w = binary_image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    # Valid box check
    if x2 <= x1 or y2 <= y1:
        return "", 0.0

    roi_original = original_bgr_image[y1:y2, x1:x2]
    roi_binary = binary_image[y1:y2, x1:x2]

    if roi_original.size == 0:
        return "", 0.0

    # Step 1: Detect text type using handwriting detector on ORIGINAL ROI
    text_type, detection_confidence = _detect_text_type(roi_original)

    # Step 2: Route to appropriate OCR engine
    if text_type == 'handwritten' and detection_confidence > 0.55:
        # Lower threshold to catch more handwriting
        print(f"    - Text classified as HANDWRITTEN (conf: {detection_confidence:.2f}). Routing to TrOCR...")

        # ADAPTIVE: Use special preprocessing for handwriting (soft denoising, no binarization)
        handwriting_ready_img = image_utils.preprocess_for_handwriting(roi_original)

        if 'TrOCRProcessor' in globals():
            trocr_text = ocr_with_trocr(handwriting_ready_img)
            if trocr_text and len(trocr_text.strip()) > 0:
                # Assume high confidence for TrOCR if text is found
                return trocr_text, 85.0

        # Fallback to PaddleOCR
        if 'paddleocr' in globals():
            paddle_text = ocr_with_paddle(handwriting_ready_img)
            if paddle_text and len(paddle_text.strip()) > 0:
                return paddle_text, 75.0

        # Last resort: Tesseract
        return ocr_with_tesseract(roi_binary, lang=lang, box=box)

    elif text_type == 'printed' and detection_confidence > 0.6:
        # Printed text -> Use Tesseract on binary image (standard pipeline)
        tess_text, tess_conf = ocr_with_tesseract(roi_binary, lang=lang, box=box)
        quality = _calculate_text_quality_score(tess_text, tess_conf)

        if quality > 60.0:
            return tess_text, tess_conf

        # If Tesseract quality is poor, try PaddleOCR on original image
        if quality < 50.0 and 'paddleocr' in globals():
            paddle_text = ocr_with_paddle(roi_original)
            if paddle_text:
                paddle_quality = _calculate_text_quality_score(paddle_text, 75.0)
                if paddle_quality > quality:
                    return paddle_text, 75.0

        return tess_text, tess_conf

    else:
        # Uncertain classification -> Fall back to smart OCR
        return ocr_smart(original_bgr_image, binary_image, box, lang)


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
    except Exception:
        return ""

def ocr_with_trocr(roi_bgr):
    """
    Runs TrOCR model on the given ROI with LINE SEGMENTATION.
    """
    _initialize_trocr()
    if _trocr_model_instance is None or _trocr_processor_instance is None:
        return ""
    
    # 1. SPLIT BLOCK INTO LINES
    # This prevents the "gibberish" output when TrOCR tries to read multiple lines at once
    line_images = split_text_block_into_lines(roi_bgr)
    full_text_parts = []
    
    for line_img in line_images:
        if line_img.shape[0] < 5 or line_img.shape[1] < 5:
            continue

        rgb_image = cv2.cvtColor(line_img, cv2.COLOR_BGR2RGB)
        try:
            pixel_values = _trocr_processor_instance(images=rgb_image, return_tensors="pt").pixel_values.to("cuda")

            # More deterministic generation to reduce hallucinations
            generated_ids = _trocr_model_instance.generate(
                pixel_values,
                max_new_tokens=64,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
                temperature=0.1,
                do_sample=True
            )
            generated_text = _trocr_processor_instance.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # Post-processing filter to discard garbage outputs
            clean_text = generated_text.strip()
            if clean_text:
                # Filter out very short non-alphanumeric text (likely hallucinations)
                has_alphanumeric = any(c.isalnum() for c in clean_text)
                if len(clean_text) < 2 and not has_alphanumeric:
                    continue  # Discard this garbage output
                full_text_parts.append(clean_text)

        except Exception as e:
            print(f"    - Error during TrOCR line inference: {e}")
            continue

    return _clean_text(" ".join(full_text_parts))

def ocr_smart(original_bgr_image, binary_image, box, lang=DEFAULT_OCR_LANG):
    """
    Applies 3-stage smart OCR: Tesseract -> PaddleOCR -> TrOCR.
    """
    x1, y1, x2, y2 = map(int, box)
    
    h, w = binary_image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    roi_binary = binary_image[y1:y2, x1:x2]
    if roi_binary.size == 0:
        return "", 0.0

    # --- Stage 1: Tesseract ---
    tess_text, tess_conf = ocr_with_tesseract(roi_binary, lang=lang, box=box)
    tess_quality = _calculate_text_quality_score(tess_text, tess_conf)

    # Check if result is good enough
    if tess_quality > 80.0:
        return tess_text, tess_conf

    best_text = tess_text
    best_conf = tess_conf
    best_quality = tess_quality

    roi_original = original_bgr_image[y1:y2, x1:x2]

    # Suspicion check
    is_fragmented = _is_likely_handwritten(tess_text, tess_conf)
    should_try_alternatives = (tess_conf < 60.0 or tess_quality < 50.0 or is_fragmented)

    # --- Stage 2: PaddleOCR ---
    if should_try_alternatives and 'paddleocr' in globals():
        paddle_text = ocr_with_paddle(roi_original)
        if paddle_text:
            paddle_quality = _calculate_text_quality_score(paddle_text, 75.0)
            if paddle_quality > (best_quality + 10): 
                best_text = paddle_text
                best_conf = 75.0
                best_quality = paddle_quality

    # --- Stage 3: TrOCR (Handwriting Expert) ---
    is_still_bad = best_quality < 45.0 or _is_likely_handwritten(best_text, best_conf)
    
    if is_still_bad and 'TrOCRProcessor' in globals():
        # Use adaptive processing
        hw_roi = image_utils.preprocess_for_handwriting(roi_original)
        trocr_text = ocr_with_trocr(hw_roi)
        if trocr_text:
            trocr_quality = _calculate_text_quality_score(trocr_text, 85.0)
            if trocr_quality > best_quality:
                best_text = trocr_text
                best_conf = 85.0
    
    if best_quality < 20.0:
        return "", 0.0

    return best_text, best_conf