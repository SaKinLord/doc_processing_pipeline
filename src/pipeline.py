# src/pipeline.py

import os
import cv2
import numpy as np
import pandas as pd
import pytesseract
from langdetect import detect, LangDetectException, detect_langs
from . import (
    config,
    image_utils,
    layout_detection,
    ocr,
    table_extraction,
    figure_extraction,
    field_extraction
)

from .description_generator import DescriptionGenerator

class DocumentProcessor:
    """
    End-to-end pipeline class for processing a single document page.
    """
    _description_generator = None
    
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
        self.language_override = language_override # UPDATED
        
        # Data to be generated during processing
        self.bgr_image = None
        self.bgr_image_no_lines = None
        self.binary_image = None
        self.binarized_inv = None
        self.h_lines = []
        self.v_lines = []
        self.height = 0
        self.width = 0
        self.detected_lang_tess = config.DEFAULT_OCR_LANG
        
        # Output data
        self.elements = []
        self.metadata = {}

    def _prepare_image(self):
        """Loads the image and applies preprocessing steps."""
        print(f"  [1/8] Preparing image for page {self.page_num}...")
        if self.file_path:
            self.bgr_image = image_utils.load_image_as_bgr(self.file_path)
        else:
            self.bgr_image = image_utils.load_image_as_bgr(self.image_object)

        if self.bgr_image is None:
            raise ValueError("Image could not be loaded.")
            
        self.height, self.width, _ = self.bgr_image.shape
        # UPDATED: Store binarized image within the class
        self.binary_image = image_utils.preprocess_for_ocr(self.bgr_image)
        self.binarized_inv = 255 - self.binary_image

    def _detect_and_remove_lines(self):
        """Finds lines in the image and creates a new image with those lines removed."""
        print(f"  [2/8] Detecting and removing lines for layout analysis...")
        self.h_lines, self.v_lines = table_extraction.find_lines(self.binarized_inv)
        self.bgr_image_no_lines = image_utils.remove_lines(self.bgr_image, self.h_lines, self.v_lines)

    def _detect_language(self):
        """Detects the document's language with a quick OCR scan."""
        print(f"  [3/8] Detecting language...")
        
        # UPDATED: If language is provided externally, skip detection.
        if self.language_override:
            self.detected_lang_tess = self.language_override
            lang_code_map_rev = {'eng': 'en', 'tur': 'tr', 'deu': 'de', 'fra': 'fr', 'spa': 'es'}
            self.metadata['language'] = lang_code_map_rev.get(self.detected_lang_tess, 'en')
            print(f"    - Using language override: '{self.metadata['language']}' -> Tesseract lang: '{self.detected_lang_tess}'")
            return

        try:
            initial_text = pytesseract.image_to_string(self.bgr_image_no_lines, lang='eng')
            
            if not initial_text or len(initial_text.strip()) < 50:
                print("    - Not enough text to reliably detect language. Using default.")
                self.metadata['language'] = config.DEFAULT_OCR_LANG.split('+')[0]
                return

            # UPDATED: Use detect_langs to get probabilities
            detected_langs = detect_langs(initial_text)
            if not detected_langs:
                raise LangDetectException("No language detected")

            top_lang = detected_langs[0]
            lang_code = top_lang.lang
            lang_prob = top_lang.prob
            
            print(f"    - Candidate language: '{lang_code}' with confidence {lang_prob:.2f}")

            # Confidence threshold check
            if lang_prob < config.LANG_DETECT_CONFIDENCE_THRESHOLD:
                print(f"    - Confidence below threshold ({config.LANG_DETECT_CONFIDENCE_THRESHOLD}). Using default language.")
                self.metadata['language'] = config.DEFAULT_OCR_LANG.split('+')[0]
                return

            LANG_MAP = {'en': 'eng', 'tr': 'tur', 'de': 'deu', 'fr': 'fra', 'es': 'spa', 'id': 'ind'} # 'id' added
            self.detected_lang_tess = LANG_MAP.get(lang_code, 'eng')
            self.metadata['language'] = lang_code
            print(f"    - Language confirmed: '{lang_code}' -> Using Tesseract lang: '{self.detected_lang_tess}'")

        except (LangDetectException, pytesseract.TesseractError) as e:
            print(f"    - Language detection failed ({e}). Using default language.")
            self.metadata['language'] = config.DEFAULT_OCR_LANG.split('+')[0]

    
    def _extract_text_blocks(self):
        """Finds text blocks on the line-cleaned image with the detected language."""
        print("  [4/8] Performing layout analysis and OCR on clean image...")
        
        detected_elements = layout_detection.detect_text_blocks(self.bgr_image_no_lines, lang=self.detected_lang_tess)
        
        temp_elements = []
        for i, block_data in enumerate(detected_elements):
            text = block_data['content']
            box = block_data['bounding_box']

            # UPDATED: Provide both original and binarized images to ocr_smart
            text, conf = ocr.ocr_smart(
                original_bgr_image=self.bgr_image,
                binary_image=self.binary_image,
                box=box,
                lang=self.detected_lang_tess
            )

            temp_elements.append({
                "id": f"textblock_{i+1}",
                "type": "TextBlock",
                "bounding_box": box,
                "content": text,
                "meta": {"ocr_conf": round(conf, 2)}
            })
        
        self.elements.extend(layout_detection.reorder_blocks_by_columns(temp_elements))

    def _extract_tables(self):
        """Hybrid table extraction followed by natural language description generation."""
        print("  [5/8] Extracting tables...")
        
        merged_h = table_extraction.merge_lines(self.h_lines, axis='h', tol=config.TABLE_LINE_MERGE_TOLERANCE)
        merged_v = table_extraction.merge_lines(self.v_lines, axis='v', tol=config.TABLE_LINE_MERGE_TOLERANCE)

        found_lined_table = False
        if len(merged_h) >= 2 and len(merged_v) >= 2:
            cells = table_extraction.lines_to_cells(merged_h, merged_v)
            
            text_blocks = [el for el in self.elements if el['type'] == 'TextBlock']
            if not table_extraction.validate_table_structure(cells, text_blocks):
                print("    - Detected grid is likely a form/layout structure, not a data table. Skipping.")
            elif len(cells) >= 4:
                print(f"    - Found a potential lined table with {len(cells)} cells. Validating content...")
                
                table_cells_data = []
                for cell_box in cells:
                    text, _ = ocr.ocr_smart(
                        original_bgr_image=self.bgr_image,
                        binary_image=self.binary_image, 
                        box=cell_box, 
                        lang=self.detected_lang_tess
                    )
                    table_cells_data.append({"bbox": cell_box, "text": text})

                grid_map, num_rows, num_cols = table_extraction.index_cells(table_cells_data)
                
                if grid_map and num_rows >= 2 and num_cols >= 2:
                    df = table_extraction.grid_to_dataframe(grid_map, num_rows, num_cols)
                    
                    table_filename = f"{os.path.splitext(self.file_name)[0]}_p{self.page_num}_table_lined_1.csv"
                    csv_path = os.path.join(config.TABLES_DIR, table_filename)
                    df.to_csv(csv_path, index=False)

                    self.elements.append({
                        "id": "table_lined_1",
                        "type": "Table",
                        "shape": {"rows": num_rows, "cols": num_cols},
                        "cells": table_cells_data,
                        "export": {"csv_path": csv_path}
                    })
                    found_lined_table = True
                else:
                    print("    - Lined cells detected but grid structure is insufficient.")

        if not found_lined_table:
            print("    - No valid lined table found. Checking for borderless tables...")
            text_blocks_for_table = [el for el in self.elements if el['type'] == 'TextBlock']
            table_candidate_blocks = table_extraction.find_borderless_table_blocks(text_blocks_for_table)

            if len(table_candidate_blocks) >= 4:
                print(f"    - Found a potential borderless table with {len(table_candidate_blocks)} blocks.")
                
                table_cells_data = []
                for block in table_candidate_blocks:
                    table_cells_data.append({"bbox": block['bounding_box'], "text": block['content']})

                grid_map, num_rows, num_cols = table_extraction.index_cells(table_cells_data)
                
                if grid_map and num_rows >= 2 and num_cols >= 2:
                    df = table_extraction.grid_to_dataframe(grid_map, num_rows, num_cols)
                    
                    table_filename = f"{os.path.splitext(self.file_name)[0]}_p{self.page_num}_table_borderless_1.csv"
                    csv_path = os.path.join(config.TABLES_DIR, table_filename)
                    df.to_csv(csv_path, index=False)

                    self.elements.append({
                        "id": "table_borderless_1",
                        "type": "Table",
                        "shape": {"rows": num_rows, "cols": num_cols},
                        "cells": table_cells_data,
                        "export": {"csv_path": csv_path}
                    })
                    print(f"    - Borderless table saved to {csv_path}")
                else:
                    print("    - Borderless cells detected but grid structure is insufficient.")
            else:
                print("    - No borderless table candidates found.")
            
        tables_in_elements = [el for el in self.elements if el['type'] == 'Table']
        if tables_in_elements:
            # Initialize generator only when needed and only once
            if DocumentProcessor._description_generator is None:
                DocumentProcessor._description_generator = DescriptionGenerator()
            
            if DocumentProcessor._description_generator.pipe:
                print("    - Generating natural language descriptions for tables...")
                for table_element in tables_in_elements:
                    csv_path = table_element.get("export", {}).get("csv_path")
                    if csv_path and os.path.exists(csv_path):
                        df = pd.read_csv(csv_path)
                        description = DocumentProcessor._description_generator.generate_for_table(df)
                        table_element['description'] = description    

    def _extract_figures(self):
        """Figure extraction followed by natural language description generation."""
        print("  [6/8] Extracting figures...")
        figure_boxes = figure_extraction.find_figure_candidates(self.binarized_inv)
        text_blocks = [el for el in self.elements if el['type'] == 'TextBlock']

        for i, box in enumerate(figure_boxes):
            box_area = (box[2] - box[0]) * (box[3] - box[1])
            if box_area == 0: continue
            
            text_overlap_area = 0
            intersecting_tbs = []
            for tb in text_blocks:
                intersection_x1 = max(box[0], tb['bounding_box'][0])
                intersection_y1 = max(box[1], tb['bounding_box'][1])
                intersection_x2 = min(box[2], tb['bounding_box'][2])
                intersection_y2 = min(box[3], tb['bounding_box'][3])
                
                if intersection_x2 > intersection_x1 and intersection_y2 > intersection_y1:
                    intersecting_tbs.append(tb)
                    text_overlap_area += (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)
            
            if (text_overlap_area / box_area) > 0.5:
                continue

            x1, y1, x2, y2 = map(int, box)
            roi_h, roi_w = y2 - y1, x2 - x1
            if roi_h == 0 or roi_w == 0: continue

            text_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
            for tb in intersecting_tbs:
                rel_x1 = int(max(0, tb['bounding_box'][0] - x1))
                rel_y1 = int(max(0, tb['bounding_box'][1] - y1))
                rel_x2 = int(min(roi_w, tb['bounding_box'][2] - x1))
                rel_y2 = int(min(roi_h, tb['bounding_box'][3] - y1))
                cv2.rectangle(text_mask, (rel_x1, rel_y1), (rel_x2, rel_y2), 255, -1)

            bin_roi = self.binarized_inv[y1:y2, x1:x2]
            roi_no_text = cv2.bitwise_and(bin_roi, cv2.bitwise_not(text_mask))
            
            non_text_content_density = np.sum(roi_no_text > 0) / roi_no_text.size
            if non_text_content_density < 0.05:
                print(f"    - Skipping figure candidate {i+1} due to low content density ({non_text_content_density:.2f}) after text removal.")
                continue

            roi = self.bgr_image[y1:y2, x1:x2]
            kind = figure_extraction.classify_figure_kind(roi)
            caption_block = figure_extraction.find_nearest_caption(box, text_blocks)
            caption = caption_block['content'] if caption_block else ""
            
            self.elements.append({
                "id": f"figure_{i+1}",
                "type": "Figure",
                "bounding_box": box,
                "kind": kind,
                "caption": caption
            })

        figures_in_elements = [el for el in self.elements if el['type'] == 'Figure']
        if figures_in_elements:
            # Initialize generator only when needed and only once
            if DocumentProcessor._description_generator is None:
                DocumentProcessor._description_generator = DescriptionGenerator()

            if DocumentProcessor._description_generator.pipe:
                print("    - Generating natural language descriptions for figures...")
                for fig_element in figures_in_elements:
                    kind = fig_element.get("kind", "figure")
                    caption = fig_element.get("caption", "")
                    description = DocumentProcessor._description_generator.generate_for_figure(kind, caption)
                    fig_element['description'] = description

    def _extract_fields(self):
        """Finds key-value pairs (form fields)."""
        print("  [7/8] Extracting key-value fields...")
        text_blocks = [el for el in self.elements if el['type'] == 'TextBlock']
        self.metadata['fields'] = field_extraction.find_fields_by_location(text_blocks)

    def process(self):
        """Runs all steps in sequence and returns the final JSON output."""
        try:
            self._prepare_image()
            self._detect_and_remove_lines()
            self._detect_language()
            self._extract_text_blocks()
            self._extract_tables()
            self._extract_figures()
            self._extract_fields()
            print(f"  [8/8] Finalizing output for {self.file_name} (Page {self.page_num})")
            return self.to_json_structure()
        except Exception as e:
            print(f"  [ERROR] Failed to process page {self.page_num} of {self.file_name}: {e}")
            return None

    def to_json_structure(self):
        """Returns the processed data in standard JSON format."""
        return {
            "page": self.page_num,
            "page_width": self.width,
            "page_height": self.height,
            "elements": self.elements,
            "metadata": self.metadata
        }