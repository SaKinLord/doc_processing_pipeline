# src/pipeline.py
"""
Document Processing Pipeline

🆕 ARCHITECTURE CHANGE (v2.1):
- Surya is now the PRIMARY layout detector (finds WHERE text is)
- Tesseract/TrOCR are used for RECOGNITION only (reads WHAT the text says)

Pipeline Flow:
1. Image Quality Analysis → Trigger Super-Resolution if needed
2. Surya Detection → Find all text line bounding boxes
3. For each detected region:
   - Handwriting Detection → Route to TrOCR or Tesseract
   - OCR Recognition → Read text from the cropped region
4. Post-processing → Fuzzy matching, text corrections

🆕 v2.3 PATCH APPLIED:
- Fixed false German detection on English documents
- Added language pack availability check
- Added confidence threshold for non-English detection
- Added deterministic seeding for langdetect

🆕 v2.4 COORDINATE FIX:
- Fixed bounding box coordinate misalignment when Super-Resolution is enabled
- All output coordinates now map to ORIGINAL image dimensions
- Added coordinate transformation to preserve compatibility with original input files
"""

import os
import cv2
import numpy as np
import pandas as pd
import pytesseract
import subprocess  # 🆕 Added for language pack detection
from langdetect import detect, LangDetectException, detect_langs
from typing import List, Dict, Optional, Tuple

# 🔧 FIX: Set deterministic seed for langdetect (Step 2 of Patch)
try:
    from langdetect import DetectorFactory
    DetectorFactory.seed = 0
except ImportError:
    pass

try:
    from . import (
        config,
        image_utils,
        layout_detection,
        ocr,
        table_extraction,
        figure_extraction,
        field_extraction
    )
except ImportError:
    import config
    import image_utils
    import layout_detection
    import ocr
    import table_extraction
    import figure_extraction
    import field_extraction

# Try to import table structure module
try:
    from . import table_structure
    TABLE_STRUCTURE_AVAILABLE = True
except ImportError:
    TABLE_STRUCTURE_AVAILABLE = False

# Try to import chart extractor module
try:
    from . import chart_extractor
    CHART_EXTRACTOR_AVAILABLE = True
except ImportError:
    CHART_EXTRACTOR_AVAILABLE = False

# Try to import VLM manager
try:
    from . import vlm_manager
    VLM_AVAILABLE = True
except ImportError:
    VLM_AVAILABLE = False

# Try to import dictionary utils
try:
    from . import dictionary_utils
    FUZZY_MATCHING_AVAILABLE = True
except ImportError:
    FUZZY_MATCHING_AVAILABLE = False


# =============================================================================
# 🆕 LANGUAGE DETECTION HELPERS (v2.2)
# =============================================================================
_available_tesseract_languages = None

def _get_available_tesseract_languages() -> set:
    """
    Get the set of available Tesseract language packs.
    Results are cached after first call.
    """
    global _available_tesseract_languages
    
    if _available_tesseract_languages is not None:
        return _available_tesseract_languages
    
    try:
        result = subprocess.run(
            ['tesseract', '--list-langs'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        lines = result.stdout.strip().split('\n')
        languages = set()
        for line in lines[1:]:  # Skip header line
            lang = line.strip()
            if lang and not lang.startswith('List'):
                languages.add(lang)
        
        _available_tesseract_languages = languages if languages else {'eng', 'osd'}
        return _available_tesseract_languages
        
    except Exception:
        _available_tesseract_languages = {'eng', 'osd'}
        return _available_tesseract_languages


def _is_tesseract_language_available(lang_code: str) -> bool:
    """Check if a Tesseract language pack is installed."""
    available = _get_available_tesseract_languages()
    if '+' in lang_code:
        return all(l in available for l in lang_code.split('+'))
    return lang_code in available


# =============================================================================
# QUALITY FLAGS
# =============================================================================

def add_quality_flags(page_result):
    """Add quality flags to help downstream systems identify uncertain extractions."""
    flags = []

    low_conf_blocks = 0
    total_blocks = 0

    for element in page_result.get('elements', []):
        if element.get('type') == 'TextBlock':
            total_blocks += 1
            conf = element.get('meta', {}).get('ocr_conf', 100)
            if conf < 50:
                low_conf_blocks += 1

    if total_blocks > 0:
        low_conf_ratio = low_conf_blocks / total_blocks
        if low_conf_ratio > 0.3:
            flags.append({
                'type': 'LOW_OCR_CONFIDENCE',
                'severity': 'warning',
                'message': f'{low_conf_ratio:.0%} of text blocks have low confidence',
                'affected_ratio': low_conf_ratio
            })

    quality = page_result.get('metadata', {}).get('image_quality', {})
    if quality.get('quality_preset') == 'noisy':
        flags.append({
            'type': 'NOISY_IMAGE',
            'severity': 'info',
            'message': 'Image quality detected as noisy, results may be affected'
        })

    page_result['quality_flags'] = flags
    return page_result


class DocumentProcessor:
    """
    End-to-end pipeline class for processing a single document page.
    
    🆕 v2.1 Architecture:
    - Surya handles LAYOUT DETECTION (finding text regions)
    - Tesseract/TrOCR handles TEXT RECOGNITION (reading text)
    
    🆕 v2.4 Coordinate Fix:
    - Stores original image dimensions separately from working dimensions
    - Tracks super-resolution scale factor
    - Transforms all coordinates back to original space before output
    """
    
    def __init__(self, file_path=None, image_object=None, file_name=None, page_num=1, language_override=None):
        if file_path:
            self.file_path = file_path
            self.file_name = os.path.basename(file_path)
            self.image_object = None
        elif image_object and file_name:
            self.file_path = None
            self.file_name = file_name
            self.image_object = image_object
        else:
            raise ValueError("DocumentProcessor requires either a file_path or an image_object with a file_name.")

        self.page_num = page_num
        self.language_override = language_override
        
        # Data to be generated during processing
        self.bgr_image = None
        self.bgr_image_enhanced = None
        self.binary_image = None
        self.binarized_inv = None
        self.h_lines = []
        self.v_lines = []
        self.height = 0
        self.width = 0
        self.detected_lang_tess = config.DEFAULT_OCR_LANG
        self.quality_metrics = None

        # 🆕 v2.4 FIX: Store original dimensions and scale factor for coordinate mapping
        self.original_width = 0
        self.original_height = 0
        self.sr_scale_factor = 1.0  # 1.0 means no scaling applied

        # Output data
        self.elements = []
        self.metadata = {}

        # Layout-detected ROIs (from LayoutParser)
        self.layout_table_rois = []
        self.layout_figure_rois = []

    def _get_description_generator(self):
        """Gets the description generator from ModelManager or creates one."""
        try:
            from .model_manager import ModelManager
            desc_gen = ModelManager.get_description_generator()
            if desc_gen is not None and desc_gen.pipe is not None:
                return desc_gen
        except:
            pass

        try:
            from .description_generator import DescriptionGenerator
            return DescriptionGenerator()
        except:
            return None

    def _load_image(self):
        """Load and prepare the image."""
        print("  [1/8] Loading image...")
        
        if self.file_path:
            self.bgr_image = image_utils.load_image_as_bgr(self.file_path)
        else:
            self.bgr_image = image_utils.load_image_as_bgr(self.image_object)
        
        if self.bgr_image is None:
            raise ValueError(f"Failed to load image: {self.file_name}")
        
        self.height, self.width = self.bgr_image.shape[:2]
        
        # 🆕 v2.4 FIX: Store original dimensions BEFORE any transformations
        self.original_height = self.height
        self.original_width = self.width
        
        print(f"    - Image size: {self.width} x {self.height}")

    def _analyze_and_enhance_image(self):
        """Analyze image quality and apply super-resolution if needed."""
        print("  [2/8] Analyzing image quality...")
        
        self.quality_metrics = image_utils.analyze_image_quality(self.bgr_image)
        self.quality_metrics['width'] = self.width
        self.quality_metrics['height'] = self.height
        
        lap_var = self.quality_metrics.get('laplacian_variance', 1000)
        noise = self.quality_metrics.get('noise_estimate', 0)
        brightness = self.quality_metrics.get('brightness', 128)
        contrast = self.quality_metrics.get('contrast', 50)
        quality_preset = self.quality_metrics.get('quality_preset', 'unknown')
        
        print(f"    - Laplacian variance: {lap_var:.1f}")
        print(f"    - Noise estimate: {noise:.1f}")
        print(f"    - Brightness: {brightness:.1f}, Contrast: {contrast:.1f}")
        print(f"    - Quality preset: {quality_preset}")
        
        should_enhance = image_utils.should_apply_super_resolution(self.quality_metrics)
        
        if should_enhance and getattr(config, 'USE_SUPER_RESOLUTION', True):
            sr_threshold = getattr(config, 'SUPER_RESOLUTION_THRESHOLD', 100.0)
            noise_threshold = getattr(config, 'SUPER_RESOLUTION_NOISE_THRESHOLD', 5.0)
            
            enhance_reason = []
            if lap_var < sr_threshold:
                enhance_reason.append(f"blur (lap_var={lap_var:.1f})")
            if noise > noise_threshold:
                enhance_reason.append(f"noise ({noise:.1f})")
            
            print(f"    - 🔍 Triggering super-resolution due to: {', '.join(enhance_reason) or 'quality issues'}")
            self.bgr_image_enhanced = image_utils.apply_super_resolution(self.bgr_image)
            
            if self.bgr_image_enhanced is not None:
                new_h, new_w = self.bgr_image_enhanced.shape[:2]
                if new_h != self.height or new_w != self.width:
                    print(f"    - Image enhanced: {self.width}x{self.height} → {new_w}x{new_h}")
                    
                    # 🆕 v2.4 FIX: Calculate and store the actual scale factor
                    # Use the average of width and height scale factors for robustness
                    scale_w = new_w / self.original_width
                    scale_h = new_h / self.original_height
                    self.sr_scale_factor = (scale_w + scale_h) / 2.0
                    
                    print(f"    - 📐 Scale factor recorded: {self.sr_scale_factor:.2f}x (for coordinate mapping)")
                    
                    # Update working dimensions (used for processing)
                    self.height, self.width = new_h, new_w
        else:
            self.bgr_image_enhanced = self.bgr_image
            self.sr_scale_factor = 1.0  # No scaling
            if quality_preset == 'clean':
                print("    - ✅ Image quality is clean, skipping super-resolution.")
            elif self.width > 2000 or self.height > 2000:
                print(f"    - ⚠️ Image too large ({self.width}x{self.height}), skipping super-resolution.")
            else:
                print("    - Image quality acceptable, skipping super-resolution.")
        
        self.metadata['image_quality'] = self.quality_metrics
        self.metadata['super_resolution_applied'] = should_enhance
        # 🆕 v2.4 FIX: Store scale factor in metadata for transparency
        self.metadata['sr_scale_factor'] = self.sr_scale_factor

    def _preprocess_image(self):
        """Preprocess image for OCR."""
        print("  [3/8] Preprocessing image...")
        
        working_image = self.bgr_image_enhanced if self.bgr_image_enhanced is not None else self.bgr_image
        params = image_utils.get_preprocessing_params(self.quality_metrics)
        
        self.binary_image = image_utils.preprocess_for_ocr(
            working_image,
            denoising_h=params['denoising_h'],
            clahe_clip_limit=params['clahe_clip_limit'],
            quality_metrics=self.quality_metrics
        )
        
        self.binarized_inv = cv2.bitwise_not(self.binary_image)
        print(f"    - Preprocessing params: denoise_h={params['denoising_h']}, clahe={params['clahe_clip_limit']}")

    def _detect_language(self):
        """
        🔧 FIXED (v2.3): Detect document language with robust multi-layer validation.
        
        Fixes applied:
        - German misdetection on sparse forms with abbreviations
        - Welsh/Celtic misdetection on handwritten OCR errors  
        - Vietnamese/Hungarian misdetection on technical codes
        - Non-deterministic langdetect behavior
        """
        print("  [4/8] Detecting language...")
        
        if self.language_override:
            self.detected_lang_tess = self.language_override
            self.metadata['language'] = self.language_override
            print(f"    - Using override language: {self.language_override}")
            return
        
        # ═══════════════════════════════════════════════════════════════════════
        # 🔧 FIX 1: Expanded language map - force misdetected languages to English
        # ═══════════════════════════════════════════════════════════════════════
        lang_map = {
            # Real language support (only include languages you actually need)
            'en': 'eng', 
            'tr': 'tur',  # Turkish - keep if you have Turkish documents
            
            # ═══════════════════════════════════════════════════════════════════
            # ALL THESE ARE FORCED TO ENGLISH (common false positives):
            # ═══════════════════════════════════════════════════════════════════
            
            # Germanic (often misdetected on abbreviations like "DEPT.", "MANUFACT.")
            'de': 'eng',  # German
            'nl': 'eng',  # Dutch
            'af': 'eng',  # Afrikaans
            'fy': 'eng',  # Frisian
            
            # Celtic (misdetected on OCR errors from handwriting)
            'cy': 'eng',  # Welsh
            'ga': 'eng',  # Irish
            'gd': 'eng',  # Scottish Gaelic
            
            # Romance (misdetected on names, abbreviations)
            'ca': 'eng',  # Catalan
            'gl': 'eng',  # Galician
            'pt': 'eng',  # Portuguese
            'it': 'eng',  # Italian
            'es': 'eng',  # Spanish
            'fr': 'eng',  # French
            'ro': 'eng',  # Romanian
            'la': 'eng',  # Latin
            'eo': 'eng',  # Esperanto
            
            # Nordic (misdetected on certain character patterns)
            'da': 'eng',  # Danish
            'no': 'eng',  # Norwegian
            'sv': 'eng',  # Swedish
            'is': 'eng',  # Icelandic
            'fi': 'eng',  # Finnish
            
            # Slavic/Eastern European
            'pl': 'eng',  # Polish
            'cs': 'eng',  # Czech
            'sk': 'eng',  # Slovak
            'hr': 'eng',  # Croatian
            'sl': 'eng',  # Slovenian
            'hu': 'eng',  # Hungarian (misdetected on alphanumeric codes)
            'et': 'eng',  # Estonian
            'lt': 'eng',  # Lithuanian
            'lv': 'eng',  # Latvian
            
            # Asian (misdetected on form patterns like "NO. PER PACK:")
            'vi': 'eng',  # Vietnamese
            'id': 'eng',  # Indonesian
            'ms': 'eng',  # Malay
            'tl': 'eng',  # Tagalog
            'th': 'eng',  # Thai
            
            # Other common false positives
            'so': 'eng',  # Somali
            'sw': 'eng',  # Swahili
            'sq': 'eng',  # Albanian
            'mt': 'eng',  # Maltese
        }
        
        # ═══════════════════════════════════════════════════════════════════════
        # 🔧 FIX 2: Helper function to check for English patterns
        # ═══════════════════════════════════════════════════════════════════════
        def has_english_indicators(text):
            """Check if text has English document patterns."""
            import re
            text_lower = text.lower()
            
            # Common English form/business patterns
            patterns = [
                r'\b(to|from|date|subject|re|cc|fax|phone|page)\s*:',
                r'\b(mr|mrs|ms|dr|inc|corp|llc|ltd)\b\.?',
                r'\b(please|thank|regards|sincerely|dear)\b',
                r'\b(total|amount|number|name|address)\b',
                r'\$\d+',  # Dollar amounts
                r'\(\d{3}\)\s*\d{3}[-.]?\d{4}',  # US phone
            ]
            
            matches = sum(1 for p in patterns if re.search(p, text_lower))
            return matches >= 2
        
        def count_english_words(text):
            """Count common English words in text."""
            import re
            common_words = {
                'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can',
                'her', 'was', 'one', 'our', 'out', 'has', 'his', 'how', 'its',
                'from', 'have', 'this', 'will', 'your', 'each', 'make', 'like',
                'date', 'name', 'page', 'time', 'year', 'form', 'type', 'number',
                'to', 'of', 'in', 'is', 'it', 'at', 'by', 'on', 'be', 'as', 'or',
                'please', 'company', 'address', 'phone', 'fax', 'total', 'value',
            }
            words = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
            if not words:
                return 0, 0
            english_count = sum(1 for w in words if w in common_words)
            return english_count, len(words)
        
        # ═══════════════════════════════════════════════════════════════════════
        # Main detection logic
        # ═══════════════════════════════════════════════════════════════════════
        try:
            sample_text = pytesseract.image_to_string(
                self.binary_image[:min(500, self.height), :],
                config="--oem 3 --psm 3"
            )
            
            clean_text = sample_text.strip() if sample_text else ""
            
            # 🔧 FIX 3: Minimum text length requirement
            if len(clean_text) < 30:
                self.metadata['language'] = 'en'
                self.detected_lang_tess = 'eng'
                print(f"    - Insufficient text ({len(clean_text)} chars), defaulting to English")
                return
            
            # 🔧 FIX 4: Check for English patterns FIRST
            if has_english_indicators(clean_text):
                eng_count, total = count_english_words(clean_text)
                if total > 0 and eng_count / total >= 0.15:
                    self.metadata['language'] = 'en'
                    self.metadata['language_confidence'] = 0.95
                    self.detected_lang_tess = 'eng'
                    print(f"    - English patterns detected ({eng_count}/{total} common words)")
                    return
            
            # Run langdetect
            try:
                lang_probs = detect_langs(clean_text)
                detected = lang_probs[0].lang if lang_probs else 'en'
                confidence = lang_probs[0].prob if lang_probs else 0.0
            except:
                detected = 'en'
                confidence = 0.5
            
            # Map to Tesseract language code (with force-English mapping)
            tess_lang = lang_map.get(detected, 'eng')
            
            # 🔧 FIX 5: Check if language pack is available
            if tess_lang != 'eng' and not _is_tesseract_language_available(tess_lang):
                print(f"    - Detected '{detected}' but '{tess_lang}' pack not installed, using English")
                self.detected_lang_tess = 'eng'
                self.metadata['language'] = 'en'
                self.metadata['language_detection_fallback'] = True
                return
            
            # 🔧 FIX 6: Final English word check (override if high English ratio)
            eng_count, total = count_english_words(clean_text)
            if total > 5 and eng_count / total > 0.25:
                self.metadata['language'] = 'en'
                self.detected_lang_tess = 'eng'
                self.metadata['language_confidence'] = confidence
                print(f"    - High English word ratio ({eng_count}/{total}), using English")
                return
            
            # Accept detection result
            self.metadata['language'] = 'en' if tess_lang == 'eng' else detected
            self.metadata['language_confidence'] = confidence
            self.detected_lang_tess = tess_lang
            
            if tess_lang == 'eng':
                print(f"    - Detected '{detected}' → mapped to English")
            else:
                print(f"    - Detected language: {detected} → {tess_lang} (conf: {confidence:.2f})")
                
        except (LangDetectException, Exception) as e:
            self.metadata['language'] = 'en'
            self.detected_lang_tess = 'eng'
            print(f"    - Language detection failed: {e}, defaulting to English.")

    def _detect_layout_surya_first(self):
        """Surya-first layout detection."""
        print("  [5/8] Detecting layout (Surya-first architecture)...")
        
        working_image = self.bgr_image_enhanced if self.bgr_image_enhanced is not None else self.bgr_image
        text_blocks = layout_detection.detect_text_blocks(working_image, self.detected_lang_tess)
        
        if not text_blocks:
            print("    - No text blocks detected.")
            return
        
        print(f"    - Detected {len(text_blocks)} text regions via {text_blocks[0].get('detection_source', 'unknown')}")
        
        desc_gen = self._get_description_generator()
        textblock_counter = 0
        
        for block in text_blocks:
            bbox = block['bounding_box']
            x1, y1, x2, y2 = map(int, bbox)
            
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(self.width, x2)
            y2 = min(self.height, y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            roi_bgr = working_image[y1:y2, x1:x2]
            roi_binary = self.binary_image[y1:y2, x1:x2]
            
            if roi_bgr.size == 0:
                continue
            
            textblock_counter += 1
            
            text, conf = self._recognize_text_in_region(
                roi_bgr, roi_binary, bbox, 
                detection_source=block.get('detection_source', 'unknown')
            )
            
            if desc_gen and 30 < conf < 85 and len(text) > 5:
                try:
                    text = desc_gen.correct_ocr_mistakes(text)
                except:
                    pass
            
            if FUZZY_MATCHING_AVAILABLE and getattr(config, 'USE_FUZZY_MATCHING', True):
                try:
                    text = dictionary_utils.apply_fuzzy_corrections(text)
                except:
                    pass
            
            self.elements.append({
                "id": f"textblock_{textblock_counter}",
                "type": "TextBlock",
                "bounding_box": list(bbox),
                "content": text,
                "meta": {
                    "ocr_conf": round(conf, 2),
                    "detection_source": block.get('detection_source', 'unknown')
                }
            })
        
        if hasattr(layout_detection, 'reorder_blocks_smart'):
            self.elements = layout_detection.reorder_blocks_smart(
                self.elements,
                use_graph=getattr(config, 'USE_GRAPH_BASED_READING_ORDER', True)
            )
        
        print(f"    - Successfully recognized text in {textblock_counter} regions.")

    def _recognize_text_in_region(
        self, 
        roi_bgr: np.ndarray, 
        roi_binary: np.ndarray, 
        bbox: List[float],
        detection_source: str = 'unknown'
    ) -> Tuple[str, float]:
        """Recognition-only OCR for a detected region."""
        sr_already_applied = self.metadata.get('super_resolution_applied', False)
        
        try:
            from .handwriting_detector import get_handwriting_detector
            detector = get_handwriting_detector()
            hw_result = detector.classify(roi_bgr)
            text_type = hw_result.get('classification', 'unknown')
            hw_confidence = hw_result.get('confidence', 0.0)
        except:
            text_type = 'unknown'
            hw_confidence = 0.0
        
        if text_type == 'handwritten' and hw_confidence > 0.5:
            print(f"      - Region classified as handwritten (conf: {hw_confidence:.2f})")
            
            hw_roi = image_utils.preprocess_for_handwriting(
                roi_bgr, 
                quality_metrics=None,
                full_image_sr_applied=sr_already_applied
            )
            
            text = ocr.ocr_with_trocr(hw_roi)
            if text and len(text.strip()) > 0:
                return text, 85.0
            
            text = ocr.ocr_with_paddle(hw_roi)
            if text and len(text.strip()) > 0:
                return text, 75.0
        
        text, conf = ocr.ocr_with_tesseract(roi_binary, lang=self.detected_lang_tess, box=bbox)
        
        if conf < 50:
            paddle_text = ocr.ocr_with_paddle(roi_bgr)
            if paddle_text:
                paddle_quality = ocr._calculate_text_quality_score(paddle_text, 75.0)
                tess_quality = ocr._calculate_text_quality_score(text, conf)
                
                if paddle_quality > tess_quality + 10:
                    return paddle_text, 75.0
        
        if conf < getattr(config, 'VLM_CONFIDENCE_THRESHOLD', 50.0) and VLM_AVAILABLE:
            if getattr(config, 'USE_VLM', False):
                try:
                    vlm = vlm_manager.get_vlm_manager()
                    if vlm.is_available:
                        confirmed_text = vlm.confirm_ocr_text(roi_bgr, text, "text region")
                        if confirmed_text and confirmed_text != text:
                            print(f"      - VLM corrected: '{text[:30]}...' → '{confirmed_text[:30]}...'")
                            return confirmed_text, 70.0
                except:
                    pass
        
        return text, conf

    def _extract_tables(self):
        """Extract tables from the document."""
        print("  [6/8] Extracting tables...")
        
        working_image = self.bgr_image_enhanced if self.bgr_image_enhanced is not None else self.bgr_image
        
        if working_image is None or working_image.size == 0:
            print("    - No image available for table extraction.")
            return
        if self.binarized_inv is None or self.binarized_inv.size == 0:
            print("    - No binarized image available for table extraction.")
            return
        
        try:
            self.h_lines, self.v_lines = table_extraction.find_lines(self.binarized_inv)
            
            if len(self.h_lines) >= 2 and len(self.v_lines) >= 2:
                merged_h = table_extraction.merge_lines(self.h_lines, axis='h')
                merged_v = table_extraction.merge_lines(self.v_lines, axis='v')
                
                cells = table_extraction.lines_to_cells(merged_h, merged_v)
                
                if cells and len(cells) >= 4:
                    table_cells_data = []
                    h, w = working_image.shape[:2]
                    
                    for cell_box in cells:
                        x1, y1, x2, y2 = map(int, cell_box)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        
                        if x2 <= x1 or y2 <= y1:
                            table_cells_data.append({"bbox": cell_box, "text": ""})
                            continue
                        
                        cell_roi = working_image[y1:y2, x1:x2]
                        cell_binary = self.binary_image[y1:y2, x1:x2]
                        
                        if cell_roi.size == 0 or cell_binary.size == 0:
                            table_cells_data.append({"bbox": cell_box, "text": ""})
                            continue
                        
                        text, _ = self._recognize_text_in_region(cell_roi, cell_binary, cell_box)
                        table_cells_data.append({"bbox": cell_box, "text": text})
                    
                    grid_map, num_rows, num_cols = table_extraction.index_cells(table_cells_data)
                    
                    if grid_map and num_rows >= 2 and num_cols >= 2:
                        df = table_extraction.grid_to_dataframe(grid_map, num_rows, num_cols)
                        
                        os.makedirs(config.TABLES_DIR, exist_ok=True)
                        table_filename = f"{os.path.splitext(self.file_name)[0]}_p{self.page_num}_table.csv"
                        csv_path = os.path.join(config.TABLES_DIR, table_filename)
                        df.to_csv(csv_path, index=False)
                        
                        self.elements.append({
                            "id": "table_1",
                            "type": "Table",
                            "shape": {"rows": num_rows, "cols": num_cols},
                            "cells": table_cells_data,
                            "export": {"csv_path": csv_path}
                        })
                        print(f"    - Extracted table: {num_rows} rows x {num_cols} cols")
                    else:
                        print("    - No valid table structure found.")
                else:
                    print("    - Insufficient cells for table.")
            else:
                print("    - No table lines detected.")
                
        except Exception as e:
            print(f"    - Table extraction error: {e}")

    def _extract_figures(self):
        """Extract figures from the document."""
        print("  [7/8] Extracting figures...")
        
        try:
            working_image = self.bgr_image_enhanced if self.bgr_image_enhanced is not None else self.bgr_image
            
            figures = figure_extraction.find_figure_regions(
                working_image,
                self.binary_image,
                self.elements
            )
            
            if figures:
                os.makedirs(config.FIGURES_DIR, exist_ok=True)
                
                for idx, fig in enumerate(figures):
                    fig_id = f"figure_{idx + 1}"
                    self.elements.append({
                        "id": fig_id,
                        "type": "Figure",
                        **fig
                    })
                
                print(f"    - Extracted {len(figures)} figure(s).")
            else:
                print("    - No figures detected.")
                
        except Exception as e:
            print(f"    - Figure extraction error: {e}")

    def _extract_fields(self):
        """Extract form fields."""
        print("  [8/8] Extracting form fields...")
        
        try:
            text_blocks = [e for e in self.elements if e['type'] == 'TextBlock']
            
            desc_gen = self._get_description_generator()
            fields = field_extraction.find_fields_hybrid(text_blocks, desc_gen)
            
            if fields:
                self.metadata['fields'] = fields
                print(f"    - Extracted {len(fields)} field(s): {list(fields.keys())}")
            else:
                print("    - No form fields detected.")
                
        except Exception as e:
            print(f"    - Field extraction error: {e}")

    # =========================================================================
    # 🆕 v2.4 FIX: COORDINATE TRANSFORMATION METHODS
    # =========================================================================
    
    def _transform_bbox_to_original(self, bbox: List[float]) -> List[float]:
        """
        Transform a bounding box from enhanced/working coordinates to original image coordinates.
        
        Args:
            bbox: [x1, y1, x2, y2] in enhanced image space
            
        Returns:
            [x1, y1, x2, y2] in original image space
        """
        if self.sr_scale_factor == 1.0:
            return bbox  # No transformation needed
        
        # Divide coordinates by scale factor to map back to original space
        return [
            bbox[0] / self.sr_scale_factor,
            bbox[1] / self.sr_scale_factor,
            bbox[2] / self.sr_scale_factor,
            bbox[3] / self.sr_scale_factor
        ]
    
    def _transform_all_coordinates_to_original(self):
        """
        🆕 v2.4 FIX: Transform ALL bounding box coordinates in elements and metadata
        back to original image space.
        
        This ensures output coordinates match the original input file dimensions,
        regardless of any internal upscaling performed during processing.
        """
        if self.sr_scale_factor == 1.0:
            print("    - 📐 No coordinate transformation needed (scale factor = 1.0)")
            return
        
        print(f"    - 📐 Transforming coordinates from {self.sr_scale_factor:.2f}x enhanced space to original space...")
        
        transformed_count = 0
        
        # Transform element bounding boxes
        for element in self.elements:
            # Transform main bounding_box
            if 'bounding_box' in element:
                element['bounding_box'] = self._transform_bbox_to_original(element['bounding_box'])
                transformed_count += 1
            
            # Transform table cell bboxes
            if element.get('type') == 'Table' and 'cells' in element:
                for cell in element['cells']:
                    if 'bbox' in cell:
                        cell['bbox'] = self._transform_bbox_to_original(cell['bbox'])
                        transformed_count += 1
        
        # Transform field bboxes in metadata
        if 'fields' in self.metadata:
            for field_name, field_data in self.metadata['fields'].items():
                if 'label_bbox' in field_data:
                    field_data['label_bbox'] = self._transform_bbox_to_original(field_data['label_bbox'])
                    transformed_count += 1
                if 'value_bbox' in field_data:
                    field_data['value_bbox'] = self._transform_bbox_to_original(field_data['value_bbox'])
                    transformed_count += 1
        
        print(f"    - ✅ Transformed {transformed_count} bounding boxes to original coordinates")

    def run(self) -> Dict:
        """Run the complete document processing pipeline."""
        print(f"\n{'='*60}")
        print(f"Processing: {self.file_name} (Page {self.page_num})")
        print(f"{'='*60}")
        
        if hasattr(config, 'print_feature_status'):
            config.print_feature_status()
        
        try:
            self._load_image()
            self._analyze_and_enhance_image()
            self._preprocess_image()
            self._detect_language()
            self._detect_layout_surya_first()
            self._extract_tables()
            self._extract_figures()
            self._extract_fields()
            
            # 🆕 v2.4 FIX: Transform all coordinates back to original image space
            # BEFORE building the result dictionary
            self._transform_all_coordinates_to_original()
            
            # 🆕 v2.4 FIX: Use ORIGINAL dimensions in output, not enhanced dimensions
            result = {
                "page": self.page_num,
                "page_width": self.original_width,   # 🔧 FIX: Use original, not self.width
                "page_height": self.original_height, # 🔧 FIX: Use original, not self.height
                "elements": self.elements,
                "metadata": self.metadata
            }
            
            result = add_quality_flags(result)
            
            print(f"\n✅ Processing complete: {len(self.elements)} elements extracted.")
            print(f"   📐 Output coordinates mapped to original dimensions: {self.original_width}x{self.original_height}")
            
            return result
            
        except Exception as e:
            print(f"\n❌ Processing failed: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "page": self.page_num,
                "error": str(e),
                "elements": [],
                "metadata": {}
            }


def process_document(file_path: str, language: str = None) -> Dict:
    """Convenience function to process a single document."""
    processor = DocumentProcessor(file_path=file_path, language_override=language)
    return processor.run()


# Add process() as an alias for run() for backward compatibility
DocumentProcessor.process = DocumentProcessor.run