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
"""

import os
import cv2
import numpy as np
import pandas as pd
import pytesseract
from langdetect import detect, LangDetectException, detect_langs
from typing import List, Dict, Optional, Tuple

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
        self.bgr_image_enhanced = None  # 🆕 Super-resolution enhanced image
        self.binary_image = None
        self.binarized_inv = None
        self.h_lines = []
        self.v_lines = []
        self.height = 0
        self.width = 0
        self.detected_lang_tess = config.DEFAULT_OCR_LANG
        self.quality_metrics = None  # 🆕 Store quality metrics

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
        print(f"    - Image size: {self.width} x {self.height}")

    def _analyze_and_enhance_image(self):
        """
        🆕 ENHANCED: Analyze image quality and apply super-resolution if needed.
        
        Triggers super-resolution on:
        - Low sharpness (laplacian_variance < threshold)
        - High noise (noise_estimate > 5)  ← For fax documents
        
        Does NOT trigger on:
        - Clean images (already high quality)
        - Large images (> 2000px) - causes memory/crash issues
        """
        print("  [2/8] Analyzing image quality...")
        
        # Analyze quality
        self.quality_metrics = image_utils.analyze_image_quality(self.bgr_image)
        
        # 🆕 Add image dimensions to quality_metrics for SR decision
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
        
        # Check if SR should be applied using the centralized logic
        should_enhance = image_utils.should_apply_super_resolution(self.quality_metrics)
        
        # Apply super-resolution if needed
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
            
            # Update dimensions if image was upscaled
            if self.bgr_image_enhanced is not None:
                new_h, new_w = self.bgr_image_enhanced.shape[:2]
                if new_h != self.height or new_w != self.width:
                    print(f"    - Image enhanced: {self.width}x{self.height} → {new_w}x{new_h}")
                    self.height, self.width = new_h, new_w
        else:
            self.bgr_image_enhanced = self.bgr_image
            if quality_preset == 'clean':
                print("    - ✅ Image quality is clean, skipping super-resolution.")
            elif self.width > 2000 or self.height > 2000:
                print(f"    - ⚠️ Image too large ({self.width}x{self.height}), skipping super-resolution.")
            else:
                print("    - Image quality acceptable, skipping super-resolution.")
        
        # Store quality info in metadata
        self.metadata['image_quality'] = self.quality_metrics
        self.metadata['super_resolution_applied'] = should_enhance

    def _preprocess_image(self):
        """Preprocess image for OCR."""
        print("  [3/8] Preprocessing image...")
        
        # Use enhanced image if available
        working_image = self.bgr_image_enhanced if self.bgr_image_enhanced is not None else self.bgr_image
        
        # Get preprocessing parameters based on quality
        params = image_utils.get_preprocessing_params(self.quality_metrics)
        
        # Create binary image for OCR
        self.binary_image = image_utils.preprocess_for_ocr(
            working_image,
            denoising_h=params['denoising_h'],
            clahe_clip_limit=params['clahe_clip_limit'],
            quality_metrics=self.quality_metrics
        )
        
        # Create inverted binary for line detection
        self.binarized_inv = cv2.bitwise_not(self.binary_image)
        
        print(f"    - Preprocessing params: denoise_h={params['denoising_h']}, clahe={params['clahe_clip_limit']}")

    def _detect_language(self):
        """Detect document language."""
        print("  [4/8] Detecting language...")
        
        if self.language_override:
            self.detected_lang_tess = self.language_override
            self.metadata['language'] = self.language_override
            print(f"    - Using override language: {self.language_override}")
            return
        
        # Quick OCR sample for language detection
        try:
            sample_text = pytesseract.image_to_string(
                self.binary_image[:min(500, self.height), :],
                config="--oem 3 --psm 3"
            )
            
            if sample_text and len(sample_text.strip()) > 20:
                detected = detect(sample_text)
                self.metadata['language'] = detected
                
                # Map to Tesseract language codes
                lang_map = {'en': 'eng', 'tr': 'tur', 'de': 'deu', 'fr': 'fra', 'es': 'spa'}
                self.detected_lang_tess = lang_map.get(detected, 'eng')
                print(f"    - Detected language: {detected} → {self.detected_lang_tess}")
            else:
                self.metadata['language'] = 'en'
                print("    - Insufficient text for detection, defaulting to English.")
                
        except (LangDetectException, Exception) as e:
            self.metadata['language'] = 'en'
            print(f"    - Language detection failed: {e}, defaulting to English.")

    def _detect_layout_surya_first(self):
        """
        🆕 SURYA-FIRST LAYOUT DETECTION
        
        This is the key architectural change:
        1. Surya detects WHERE text is (bounding boxes)
        2. We crop those regions
        3. Tesseract/TrOCR reads WHAT the text says
        """
        print("  [5/8] Detecting layout (Surya-first architecture)...")
        
        # Use enhanced image for detection
        working_image = self.bgr_image_enhanced if self.bgr_image_enhanced is not None else self.bgr_image
        
        # Step A: Surya finds all text bounding boxes
        text_blocks = layout_detection.detect_text_blocks(working_image, self.detected_lang_tess)
        
        if not text_blocks:
            print("    - No text blocks detected.")
            return
        
        print(f"    - Detected {len(text_blocks)} text regions via {text_blocks[0].get('detection_source', 'unknown')}")
        
        # Get description generator for potential LLM corrections
        desc_gen = self._get_description_generator()
        
        # Step B & C: For each detected region, crop and recognize text
        textblock_counter = 0
        
        for block in text_blocks:
            bbox = block['bounding_box']
            x1, y1, x2, y2 = map(int, bbox)
            
            # Safety bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(self.width, x2)
            y2 = min(self.height, y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Crop the detected region
            roi_bgr = working_image[y1:y2, x1:x2]
            roi_binary = self.binary_image[y1:y2, x1:x2]
            
            if roi_bgr.size == 0:
                continue
            
            textblock_counter += 1
            
            # Recognize text using appropriate engine
            text, conf = self._recognize_text_in_region(
                roi_bgr, roi_binary, bbox, 
                detection_source=block.get('detection_source', 'unknown')
            )
            
            # Apply LLM correction for medium-confidence results
            if desc_gen and 30 < conf < 85 and len(text) > 5:
                try:
                    text = desc_gen.correct_ocr_mistakes(text)
                except:
                    pass
            
            # Apply fuzzy matching correction
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
        
        # Reorder blocks for reading order
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
        """
        🆕 RECOGNITION-ONLY OCR
        
        This function ONLY reads text - detection was already done by Surya.
        Routes to appropriate OCR engine based on content type.
        
        Args:
            roi_bgr: BGR cropped region
            roi_binary: Binary cropped region  
            bbox: Original bounding box
            detection_source: How the region was detected
        
        Returns:
            Tuple of (text, confidence)
        """
        # 🆕 PERFORMANCE FIX: Check if SR was already applied to the full image
        sr_already_applied = self.metadata.get('super_resolution_applied', False)
        
        # Check if handwriting detection is available
        try:
            from .handwriting_detector import get_handwriting_detector
            detector = get_handwriting_detector()
            hw_result = detector.classify(roi_bgr)
            text_type = hw_result.get('classification', 'unknown')
            hw_confidence = hw_result.get('confidence', 0.0)
        except:
            text_type = 'unknown'
            hw_confidence = 0.0
        
        # Route to appropriate OCR engine
        if text_type == 'handwritten' and hw_confidence > 0.5:
            # Use TrOCR for handwriting
            print(f"      - Region classified as handwritten (conf: {hw_confidence:.2f})")
            
            # 🆕 PERFORMANCE FIX: Pass sr_applied flag to skip redundant SR on regions
            hw_roi = image_utils.preprocess_for_handwriting(
                roi_bgr, 
                quality_metrics=None,  # Don't trigger SR decision
                full_image_sr_applied=sr_already_applied
            )
            
            # Try TrOCR
            text = ocr.ocr_with_trocr(hw_roi)
            if text and len(text.strip()) > 0:
                return text, 85.0
            
            # Fallback to PaddleOCR
            text = ocr.ocr_with_paddle(hw_roi)
            if text and len(text.strip()) > 0:
                return text, 75.0
        
        # Default: Use Tesseract for printed text
        text, conf = ocr.ocr_with_tesseract(roi_binary, lang=self.detected_lang_tess, box=bbox)
        
        # If Tesseract confidence is low, try PaddleOCR
        if conf < 50:
            paddle_text = ocr.ocr_with_paddle(roi_bgr)
            if paddle_text:
                paddle_quality = ocr._calculate_text_quality_score(paddle_text, 75.0)
                tess_quality = ocr._calculate_text_quality_score(text, conf)
                
                if paddle_quality > tess_quality + 10:
                    return paddle_text, 75.0
        
        # 🆕 VLM confirmation for very low confidence
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
        
        # Find lines for table detection
        working_image = self.bgr_image_enhanced if self.bgr_image_enhanced is not None else self.bgr_image
        
        # 🆕 Safety check: ensure we have valid images
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
                    # Extract table
                    table_cells_data = []
                    h, w = working_image.shape[:2]
                    
                    for cell_box in cells:
                        # 🆕 Bounds checking to prevent empty image errors
                        x1, y1, x2, y2 = map(int, cell_box)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        
                        # Skip invalid cells
                        if x2 <= x1 or y2 <= y1:
                            table_cells_data.append({"bbox": cell_box, "text": ""})
                            continue
                        
                        cell_roi = working_image[y1:y2, x1:x2]
                        cell_binary = self.binary_image[y1:y2, x1:x2]
                        
                        # Skip empty cells
                        if cell_roi.size == 0 or cell_binary.size == 0:
                            table_cells_data.append({"bbox": cell_box, "text": ""})
                            continue
                        
                        text, _ = self._recognize_text_in_region(
                            cell_roi,
                            cell_binary,
                            cell_box
                        )
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

    def run(self) -> Dict:
        """
        Run the complete document processing pipeline.
        
        Returns:
            Dictionary with page processing results
        """
        print(f"\n{'='*60}")
        print(f"Processing: {self.file_name} (Page {self.page_num})")
        print(f"{'='*60}")
        
        # Print feature status
        if hasattr(config, 'print_feature_status'):
            config.print_feature_status()
        
        try:
            # Step 1: Load image
            self._load_image()
            
            # Step 2: Analyze quality and enhance if needed (🆕 includes noise check)
            self._analyze_and_enhance_image()
            
            # Step 3: Preprocess
            self._preprocess_image()
            
            # Step 4: Detect language
            self._detect_language()
            
            # Step 5: 🆕 Surya-first layout detection + recognition
            self._detect_layout_surya_first()
            
            # Step 6: Extract tables
            self._extract_tables()
            
            # Step 7: Extract figures
            self._extract_figures()
            
            # Step 8: Extract fields
            self._extract_fields()
            
            # Build result
            result = {
                "page": self.page_num,
                "page_width": self.width,
                "page_height": self.height,
                "elements": self.elements,
                "metadata": self.metadata
            }
            
            # Add quality flags
            result = add_quality_flags(result)
            
            print(f"\n✅ Processing complete: {len(self.elements)} elements extracted.")
            
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
    """
    Convenience function to process a single document.
    
    Args:
        file_path: Path to document file
        language: Optional language override
    
    Returns:
        Processing results dictionary
    """
    processor = DocumentProcessor(file_path=file_path, language_override=language)
    return processor.run()


# Add process() as an alias for run() for backward compatibility
DocumentProcessor.process = DocumentProcessor.run
