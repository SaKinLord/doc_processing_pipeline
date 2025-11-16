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

# --- Handwriting Detector Integration ---
try:
    from .handwriting_detector import get_handwriting_detector
    _handwriting_detector_available = True
except ImportError:
    _handwriting_detector_available = False
    print("Warning: handwriting_detector module not available. Smart OCR routing will use heuristics only.")


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


def _calculate_text_quality_score(text, confidence=0.0):
    """
    Calculates a quality score for OCR text based on multiple heuristics.

    Returns a score from 0-100 where higher is better.
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
    # Reward longer text but cap at reasonable length
    length_score = min(15.0, len(text) / 10.0)
    score += length_score

    # 3. Fragmentation penalty (0-25 points)
    # High fragmentation (many short words) indicates OCR failure
    words = text.split()
    if len(words) > 0:
        avg_word_length = len(text.replace(' ', '')) / len(words)
        if avg_word_length >= 4.0:
            score += 25.0  # Good average word length
        elif avg_word_length >= 3.0:
            score += 15.0
        elif avg_word_length >= 2.0:
            score += 5.0
        # else: highly fragmented, no points

    # 4. Character composition (0-20 points)
    # Good text should have mostly alphanumeric characters
    if len(text) > 0:
        alnum_count = sum(1 for c in text if c.isalnum() or c.isspace())
        alnum_ratio = alnum_count / len(text)
        score += alnum_ratio * 20.0

    # 5. Noise patterns penalty (0-10 points)
    # Penalize common OCR failure patterns
    noise_penalty = 0.0

    # Single character repetitions (like "l l l l")
    single_char_pattern = re.findall(r'\b\w\b', text)
    if len(single_char_pattern) > 3:
        noise_penalty += min(5.0, len(single_char_pattern) * 0.5)

    # Excessive punctuation
    punct_count = sum(1 for c in text if c in '.,;:!?-_/\\|@#$%^&*()[]{}')
    if len(text) > 0 and punct_count / len(text) > 0.3:
        noise_penalty += 3.0

    # Very short words dominating
    if len(words) > 3:
        short_words = sum(1 for w in words if len(w) <= 2)
        if short_words / len(words) > 0.6:
            noise_penalty += 2.0

    score = max(0.0, score - noise_penalty)

    return min(100.0, score)


def _is_likely_handwritten(text, confidence):
    """
    Heuristic to determine if text is likely handwritten based on OCR results.

    Returns True if text appears to be handwritten (OCR struggled with it).
    """
    if not text or len(text.strip()) == 0:
        return False

    # High fragmentation with low confidence suggests handwriting
    words = text.split()
    if len(words) == 0:
        return False

    avg_word_length = len(text.replace(' ', '')) / len(words)

    # Highly fragmented with low confidence
    if avg_word_length < 3.0 and confidence < 60.0:
        return True

    # Many single-character "words"
    single_chars = sum(1 for w in words if len(w) == 1)
    if len(words) > 2 and single_chars / len(words) > 0.4:
        return True

    return False


def _get_psm_for_box(box):
    """Selects the most appropriate Tesseract PSM mode based on the box's aspect ratio."""
    x1, y1, x2, y2 = map(float, box)
    w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
    aspect_ratio = w / h
    if aspect_ratio > 5.0:
        return 7
    if w * h < 8000:
        return 13
    return 6


def _detect_text_type(roi_image):
    """
    Uses the handwriting detector to classify text as printed or handwritten.

    Args:
        roi_image: Input image (BGR or grayscale)

    Returns:
        tuple: (text_type, confidence) where text_type is 'printed' or 'handwritten'
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

    This is more efficient than the fallback chain approach as it:
    1. Pre-classifies text as printed or handwritten
    2. Routes directly to the best OCR engine
    3. Skips unnecessary OCR stages

    Args:
        original_bgr_image: Original BGR image
        binary_image: Binary (preprocessed) image
        box: Bounding box [x1, y1, x2, y2]
        lang: Language code for Tesseract

    Returns:
        tuple: (text, confidence)
    """
    x1, y1, x2, y2 = map(int, box)
    roi_binary = binary_image[y1:y2, x1:x2]
    roi_original = original_bgr_image[y1:y2, x1:x2]

    if roi_binary.size == 0 or roi_original.size == 0:
        return "", 0.0

    # Step 1: Detect text type using handwriting detector
    text_type, detection_confidence = _detect_text_type(roi_original)

    # Step 2: Route to appropriate OCR engine
    if text_type == 'handwritten' and detection_confidence > 0.7:
        # High confidence handwritten text -> Use TrOCR directly
        print(f"    - Text classified as HANDWRITTEN (conf: {detection_confidence:.2f}). Routing to TrOCR...")

        if 'TrOCRProcessor' in globals():
            trocr_text = ocr_with_trocr(roi_original)
            if trocr_text and len(trocr_text.strip()) > 0:
                quality = _calculate_text_quality_score(trocr_text, 85.0)
                if quality > 40.0:
                    print(f"    - TrOCR result: '{trocr_text[:50]}...' (quality: {quality:.1f})")
                    return trocr_text, 85.0

        # Fallback to PaddleOCR if TrOCR fails
        if 'paddleocr' in globals():
            paddle_text = ocr_with_paddle(roi_original)
            if paddle_text and len(paddle_text.strip()) > 0:
                return paddle_text, 75.0

        # Last resort: Tesseract
        return ocr_with_tesseract(roi_binary, lang=lang, box=box)

    elif text_type == 'printed' and detection_confidence > 0.7:
        # High confidence printed text -> Use Tesseract directly
        print(f"    - Text classified as PRINTED (conf: {detection_confidence:.2f}). Routing to Tesseract...")

        tess_text, tess_conf = ocr_with_tesseract(roi_binary, lang=lang, box=box)
        quality = _calculate_text_quality_score(tess_text, tess_conf)

        if quality > 60.0:
            return tess_text, tess_conf

        # If Tesseract quality is poor, try PaddleOCR as backup
        if quality < 50.0 and 'paddleocr' in globals():
            print(f"    - Tesseract quality low ({quality:.1f}). Trying PaddleOCR...")
            paddle_text = ocr_with_paddle(roi_original)
            if paddle_text:
                paddle_quality = _calculate_text_quality_score(paddle_text, 75.0)
                if paddle_quality > quality:
                    return paddle_text, 75.0

        return tess_text, tess_conf

    else:
        # Uncertain classification -> Fall back to smart OCR (original behavior)
        print(f"    - Text type uncertain (type: {text_type}, conf: {detection_confidence:.2f}). Using smart OCR...")
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
    Uses quality scoring and fragmentation heuristics to determine best result.
    """
    x1, y1, x2, y2 = map(int, box)
    roi_binary = binary_image[y1:y2, x1:x2]
    if roi_binary.size == 0:
        return "", 0.0

    # --- Stage 1: Tesseract (optimized for printed text) ---
    tess_text, tess_conf = ocr_with_tesseract(roi_binary, lang=lang, box=box)
    tess_quality = _calculate_text_quality_score(tess_text, tess_conf)

    # Check if Tesseract result is excellent
    if tess_quality > 80.0 and tess_conf > 85.0:
        return tess_text, tess_conf

    # Variables to store the best result
    best_text = tess_text
    best_conf = tess_conf
    best_quality = tess_quality

    roi_original = original_bgr_image[y1:y2, x1:x2]
    if roi_original.size == 0:
        return best_text, best_conf

    # Check if we should try alternative OCR engines
    is_fragmented = _is_likely_handwritten(tess_text, tess_conf)
    should_try_alternatives = (
        tess_conf < LOW_CONFIDENCE_THRESHOLD or
        tess_quality < 60.0 or
        is_fragmented or
        len(tess_text.strip()) < 3
    )

    # --- Stage 2: PaddleOCR Fallback (general purpose and handwriting) ---
    if should_try_alternatives and 'paddleocr' in globals():
        reason = "fragmented" if is_fragmented else f"low quality ({tess_quality:.1f})"
        print(f"    - Tesseract result {reason}. Trying PaddleOCR...")
        paddle_text = ocr_with_paddle(roi_original)

        if paddle_text:
            paddle_quality = _calculate_text_quality_score(paddle_text, 75.0)

            if paddle_quality > best_quality:
                print(f"    - PaddleOCR better (quality: {paddle_quality:.1f} vs {best_quality:.1f})")
                best_text = paddle_text
                best_conf = 75.0
                best_quality = paddle_quality

    # --- Stage 3: TrOCR Fallback (handwriting expert) ---
    # Only use TrOCR if text appears handwritten or quality is still poor
    is_still_fragmented = _is_likely_handwritten(best_text, best_conf)
    should_try_trocr = (
        (best_quality < 50.0 or is_still_fragmented) and
        'TrOCRProcessor' in globals()
    )

    if should_try_trocr:
        print(f"    - Result still weak or appears handwritten. Trying TrOCR expert...")
        trocr_text = ocr_with_trocr(roi_original)

        if trocr_text:
            trocr_quality = _calculate_text_quality_score(trocr_text, 85.0)

            if trocr_quality > best_quality:
                print(f"    - TrOCR better (quality: {trocr_quality:.1f} vs {best_quality:.1f})")
                best_text = trocr_text
                best_conf = 85.0
                best_quality = trocr_quality

    return best_text, best_conf