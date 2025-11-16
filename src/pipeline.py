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

class DocumentProcessor:
    """
    End-to-end pipeline class for processing a single document page.
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

        # Fall back to creating a new instance
        from .description_generator import DescriptionGenerator
        return DescriptionGenerator()

    def _extract_table_with_structure(self, h_lines, v_lines, table_id="table_1"):
        """
        Extracts a table with advanced structure analysis (merged cells, HTML/Markdown).

        Returns:
            dict with table element data or None if extraction fails
        """
        if not TABLE_STRUCTURE_AVAILABLE or not config.USE_ADVANCED_TABLE_STRUCTURE:
            return None

        try:
            # Define OCR function wrapper
            def ocr_wrapper(bgr_img, bin_img, bbox, lang):
                if config.USE_OCR_ROUTING and hasattr(ocr, 'ocr_routed'):
                    return ocr.ocr_routed(bgr_img, bin_img, bbox, lang)
                else:
                    return ocr.ocr_smart(bgr_img, bin_img, bbox, lang)

            result = table_structure.extract_table_with_structure(
                self.bgr_image,
                self.binarized_inv,
                h_lines,
                v_lines,
                ocr_wrapper,
                self.detected_lang_tess
            )

            if result is None:
                return None

            # Check if we have merged cells
            has_merged = len(result['merged_cells']) > 0

            if has_merged:
                print(f"    - Detected {len(result['merged_cells'])} merged cell region(s)")

            # Save exports
            base_filename = f"{os.path.splitext(self.file_name)[0]}_p{self.page_num}_{table_id}"

            # Save CSV
            csv_path = os.path.join(config.TABLES_DIR, f"{base_filename}.csv")
            result['dataframe'].to_csv(csv_path, index=False, header=False)

            # Save HTML
            html_path = os.path.join(config.TABLES_DIR, f"{base_filename}.html")
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(result['html'])

            # Save Markdown
            md_path = os.path.join(config.TABLES_DIR, f"{base_filename}.md")
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(result['markdown'])

            # Build cell data for JSON output
            cells_data = []
            for (row, col), cell_info in result['cell_data'].items():
                cells_data.append({
                    'row': row,
                    'col': col,
                    'bbox': cell_info['bbox'],
                    'text': cell_info['text'],
                    'rowspan': cell_info['rowspan'],
                    'colspan': cell_info['colspan'],
                    'is_merged': cell_info['is_merged']
                })

            return {
                'id': table_id,
                'type': 'Table',
                'shape': {'rows': result['num_rows'], 'cols': result['num_cols']},
                'cells': cells_data,
                'merged_cells': result['merged_cells'],
                'has_merged_cells': has_merged,
                'export': {
                    'csv_path': csv_path,
                    'html_path': html_path,
                    'markdown_path': md_path
                },
                'meta': {'structure_analysis': True}
            }

        except Exception as e:
            print(f"    - Advanced table structure analysis failed: {e}")
            return None

    def _prepare_image(self):
        """Loads the image and applies adaptive preprocessing steps."""
        print(f"  [1/8] Preparing image for page {self.page_num}...")
        if self.file_path:
            self.bgr_image = image_utils.load_image_as_bgr(self.file_path)
        else:
            self.bgr_image = image_utils.load_image_as_bgr(self.image_object)

        if self.bgr_image is None:
            raise ValueError("Image could not be loaded.")

        self.height, self.width, _ = self.bgr_image.shape

        # Analyze image quality for adaptive preprocessing
        quality_info = image_utils.analyze_image_quality(self.bgr_image)
        print(f"    - Image quality analysis:")
        print(f"      Laplacian variance: {quality_info['laplacian_variance']:.2f}")
        print(f"      Brightness: {quality_info['brightness']:.2f}, Contrast: {quality_info['contrast']:.2f}")
        print(f"      Quality preset: '{quality_info['quality_preset']}'")
        print(f"      Using denoising_h={quality_info['denoising_h']}, CLAHE clip={quality_info['clahe_clip_limit']}")

        # Store quality info in metadata
        self.metadata['image_quality'] = quality_info

        # Apply adaptive preprocessing
        self.binary_image = image_utils.preprocess_for_ocr(
            self.bgr_image,
            denoising_h=quality_info['denoising_h'],
            clahe_clip_limit=quality_info['clahe_clip_limit']
        )
        self.binarized_inv = 255 - self.binary_image

    def _detect_and_remove_lines(self):
        """Finds lines in the image and creates a new image with those lines removed."""
        print(f"  [2/8] Detecting and removing lines for layout analysis...")
        self.h_lines, self.v_lines = table_extraction.find_lines(self.binarized_inv)
        self.bgr_image_no_lines = image_utils.remove_lines(self.bgr_image, self.h_lines, self.v_lines)

    def _detect_language(self):
        """Detects the document's language with a quick OCR scan."""
        print(f"  [3/8] Detecting language...")
        
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

            detected_langs = detect_langs(initial_text)
            if not detected_langs:
                raise LangDetectException("No language detected")

            top_lang = detected_langs[0]
            lang_code = top_lang.lang
            lang_prob = top_lang.prob
            
            print(f"    - Candidate language: '{lang_code}' with confidence {lang_prob:.2f}")

            if lang_prob < config.LANG_DETECT_CONFIDENCE_THRESHOLD:
                print(f"    - Confidence below threshold ({config.LANG_DETECT_CONFIDENCE_THRESHOLD}). Using default language.")
                self.metadata['language'] = config.DEFAULT_OCR_LANG.split('+')[0]
                return

            LANG_MAP = {'en': 'eng', 'tr': 'tur', 'de': 'deu', 'fr': 'fra', 'es': 'spa', 'id': 'ind'}
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
        textblock_counter = 0

        for i, block_data in enumerate(detected_elements):
            # Check if this is from layoutparser (has 'layout_type') or tesseract
            if 'layout_type' in block_data:
                # From layoutparser - may not have content yet
                element_type = block_data.get('type', 'TextBlock')
                box = block_data['bounding_box']
                confidence = block_data.get('confidence', 0)

                # Store Table and Figure ROIs for later processing
                if element_type == 'Table':
                    self.layout_table_rois.append({
                        'bounding_box': box,
                        'confidence': confidence,
                        'layout_type': block_data.get('layout_type', 'Table')
                    })
                    print(f"    - LayoutParser detected Table ROI at {box} (conf: {confidence:.2f})")
                elif element_type == 'Figure':
                    self.layout_figure_rois.append({
                        'bounding_box': box,
                        'confidence': confidence,
                        'layout_type': block_data.get('layout_type', 'Figure')
                    })
                    print(f"    - LayoutParser detected Figure ROI at {box} (conf: {confidence:.2f})")
                else:
                    # TextBlock - perform OCR
                    textblock_counter += 1
                    # Use routed OCR if available and enabled
                    if config.USE_OCR_ROUTING and hasattr(ocr, 'ocr_routed'):
                        text, conf = ocr.ocr_routed(
                            original_bgr_image=self.bgr_image,
                            binary_image=self.binary_image,
                            box=box,
                            lang=self.detected_lang_tess
                        )
                    else:
                        text, conf = ocr.ocr_smart(
                            original_bgr_image=self.bgr_image,
                            binary_image=self.binary_image,
                            box=box,
                            lang=self.detected_lang_tess
                        )
                    temp_elements.append({
                        "id": f"textblock_{textblock_counter}",
                        "type": "TextBlock",
                        "bounding_box": box,
                        "content": text,
                        "meta": {"ocr_conf": round(conf, 2), "confidence": confidence}
                    })
            else:
                # From tesseract - already has content
                textblock_counter += 1
                text = block_data.get('content', '')
                box = block_data['bounding_box']

                # Use routed OCR if available and enabled
                if config.USE_OCR_ROUTING and hasattr(ocr, 'ocr_routed'):
                    text, conf = ocr.ocr_routed(
                        original_bgr_image=self.bgr_image,
                        binary_image=self.binary_image,
                        box=box,
                        lang=self.detected_lang_tess
                    )
                else:
                    text, conf = ocr.ocr_smart(
                        original_bgr_image=self.bgr_image,
                        binary_image=self.binary_image,
                        box=box,
                        lang=self.detected_lang_tess
                    )

                temp_elements.append({
                    "id": f"textblock_{textblock_counter}",
                    "type": "TextBlock",
                    "bounding_box": box,
                    "content": text,
                    "meta": {"ocr_conf": round(conf, 2)}
                })

        # Use smart reordering (graph-based or column-based based on config)
        if hasattr(layout_detection, 'reorder_blocks_smart'):
            reordered = layout_detection.reorder_blocks_smart(
                temp_elements,
                use_graph=config.USE_GRAPH_BASED_READING_ORDER
            )
        else:
            reordered = layout_detection.reorder_blocks_by_columns(temp_elements)
        self.elements.extend(reordered)

    def _extract_tables(self):
        """Hybrid table extraction followed by natural language description generation."""
        print("  [5/8] Extracting tables...")

        tables_found = False

        # Strategy 1: Use LayoutParser-detected Table ROIs if available
        if self.layout_table_rois:
            print(f"    - Processing {len(self.layout_table_rois)} table ROI(s) from LayoutParser...")
            for idx, table_roi in enumerate(self.layout_table_rois):
                roi_box = table_roi['bounding_box']
                x1, y1, x2, y2 = map(int, roi_box)

                # Extract ROI from images
                roi_binary_inv = self.binarized_inv[y1:y2, x1:x2]

                # Find lines within this specific ROI
                roi_h_lines, roi_v_lines = table_extraction.find_lines(roi_binary_inv)

                # Adjust line coordinates to global image space
                roi_h_lines = [(lx1 + x1, ly1 + y1, lx2 + x1, ly2 + y1) for lx1, ly1, lx2, ly2 in roi_h_lines]
                roi_v_lines = [(lx1 + x1, ly1 + y1, lx2 + x1, ly2 + y1) for lx1, ly1, lx2, ly2 in roi_v_lines]

                merged_h = table_extraction.merge_lines(roi_h_lines, axis='h', tol=config.TABLE_LINE_MERGE_TOLERANCE)
                merged_v = table_extraction.merge_lines(roi_v_lines, axis='v', tol=config.TABLE_LINE_MERGE_TOLERANCE)

                table_extracted = False

                # Try lined table extraction within ROI
                if len(merged_h) >= 2 and len(merged_v) >= 2:
                    cells = table_extraction.lines_to_cells(merged_h, merged_v)
                    text_blocks = [el for el in self.elements if el['type'] == 'TextBlock']

                    if table_extraction.validate_table_structure(cells, text_blocks) and len(cells) >= 4:
                        print(f"    - Table ROI {idx+1}: Found lined table with {len(cells)} cells")

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

                            table_filename = f"{os.path.splitext(self.file_name)[0]}_p{self.page_num}_table_lined_{idx+1}.csv"
                            csv_path = os.path.join(config.TABLES_DIR, table_filename)
                            df.to_csv(csv_path, index=False)

                            self.elements.append({
                                "id": f"table_lined_{idx+1}",
                                "type": "Table",
                                "bounding_box": roi_box,
                                "shape": {"rows": num_rows, "cols": num_cols},
                                "cells": table_cells_data,
                                "export": {"csv_path": csv_path},
                                "meta": {"source": "layoutparser", "confidence": table_roi['confidence']}
                            })
                            table_extracted = True
                            tables_found = True

                # Fallback: Try borderless table within ROI
                if not table_extracted:
                    print(f"    - Table ROI {idx+1}: No lined table, trying borderless detection...")
                    # Get text blocks that fall within this ROI
                    roi_text_blocks = []
                    for el in self.elements:
                        if el['type'] == 'TextBlock':
                            eb = el['bounding_box']
                            # Check if text block center is within ROI
                            cx = (eb[0] + eb[2]) / 2
                            cy = (eb[1] + eb[3]) / 2
                            if x1 <= cx <= x2 and y1 <= cy <= y2:
                                roi_text_blocks.append(el)

                    if len(roi_text_blocks) >= 4:
                        table_candidate_blocks = table_extraction.find_borderless_table_blocks(roi_text_blocks)

                        if len(table_candidate_blocks) >= 4:
                            table_cells_data = []
                            for block in table_candidate_blocks:
                                table_cells_data.append({"bbox": block['bounding_box'], "text": block['content']})

                            grid_map, num_rows, num_cols = table_extraction.index_cells(table_cells_data)

                            if grid_map and num_rows >= 2 and num_cols >= 2:
                                df = table_extraction.grid_to_dataframe(grid_map, num_rows, num_cols)

                                table_filename = f"{os.path.splitext(self.file_name)[0]}_p{self.page_num}_table_borderless_{idx+1}.csv"
                                csv_path = os.path.join(config.TABLES_DIR, table_filename)
                                df.to_csv(csv_path, index=False)

                                self.elements.append({
                                    "id": f"table_borderless_{idx+1}",
                                    "type": "Table",
                                    "bounding_box": roi_box,
                                    "shape": {"rows": num_rows, "cols": num_cols},
                                    "cells": table_cells_data,
                                    "export": {"csv_path": csv_path},
                                    "meta": {"source": "layoutparser", "confidence": table_roi['confidence']}
                                })
                                table_extracted = True
                                tables_found = True
                                print(f"    - Table ROI {idx+1}: Borderless table extracted")

                if not table_extracted:
                    print(f"    - Table ROI {idx+1}: Could not extract valid table structure")

        # Strategy 2: Fallback to full-page detection if no layout-detected tables
        if not tables_found:
            print("    - No LayoutParser table ROIs or extraction failed. Using full-page detection...")

            merged_h = table_extraction.merge_lines(self.h_lines, axis='h', tol=config.TABLE_LINE_MERGE_TOLERANCE)
            merged_v = table_extraction.merge_lines(self.v_lines, axis='v', tol=config.TABLE_LINE_MERGE_TOLERANCE)

            found_lined_table = False
            if len(merged_h) >= 2 and len(merged_v) >= 2:
                # Try advanced structure analysis first (handles merged cells)
                if TABLE_STRUCTURE_AVAILABLE and config.USE_ADVANCED_TABLE_STRUCTURE:
                    print("    - Attempting advanced table structure analysis...")
                    advanced_table = self._extract_table_with_structure(merged_h, merged_v, "table_structured_1")
                    if advanced_table:
                        self.elements.append(advanced_table)
                        found_lined_table = True
                        tables_found = True
                        print(f"    - Advanced structure table saved with {advanced_table['shape']['rows']}x{advanced_table['shape']['cols']} grid")

            if len(merged_h) >= 2 and len(merged_v) >= 2 and not found_lined_table:
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
                            "export": {"csv_path": csv_path},
                            "meta": {"source": "heuristic"}
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
                            "export": {"csv_path": csv_path},
                            "meta": {"source": "heuristic"}
                        })
                        print(f"    - Borderless table saved to {csv_path}")
                    else:
                        print("    - Borderless cells detected but grid structure is insufficient.")
                else:
                    print("    - No borderless table candidates found.")

        tables_in_elements = [el for el in self.elements if el['type'] == 'Table']
        if tables_in_elements:
            description_generator = self._get_description_generator()

            if description_generator and description_generator.pipe:
                print("    - Generating natural language descriptions for tables...")
                for table_element in tables_in_elements:
                    csv_path = table_element.get("export", {}).get("csv_path")
                    if csv_path and os.path.exists(csv_path):
                        df = pd.read_csv(csv_path)
                        description = description_generator.generate_for_table(df)
                        table_element['description'] = description

    def _extract_figures(self):
        """Figure extraction followed by natural language description generation."""
        print("  [6/8] Extracting figures...")

        text_blocks = [el for el in self.elements if el['type'] == 'TextBlock']
        figures_found = False
        figure_counter = 0

        # Strategy 1: Use LayoutParser-detected Figure ROIs if available
        if self.layout_figure_rois:
            print(f"    - Processing {len(self.layout_figure_rois)} figure ROI(s) from LayoutParser...")
            for idx, figure_roi in enumerate(self.layout_figure_rois):
                box = figure_roi['bounding_box']
                x1, y1, x2, y2 = map(int, box)

                roi_h, roi_w = y2 - y1, x2 - x1
                if roi_h == 0 or roi_w == 0:
                    continue

                # Classify the figure
                roi = self.bgr_image[y1:y2, x1:x2]
                kind = figure_extraction.classify_figure_kind(roi)

                # Find caption
                caption_block = figure_extraction.find_nearest_caption(box, text_blocks)
                caption = caption_block['content'] if caption_block else ""

                figure_counter += 1
                self.elements.append({
                    "id": f"figure_{figure_counter}",
                    "type": "Figure",
                    "bounding_box": box,
                    "kind": kind,
                    "caption": caption,
                    "meta": {"source": "layoutparser", "confidence": figure_roi['confidence']}
                })
                figures_found = True
                print(f"    - Figure ROI {idx+1}: Classified as '{kind}'")

        # Strategy 2: Fallback to full-page morphological detection if no layout-detected figures
        if not figures_found:
            print("    - No LayoutParser figure ROIs. Using full-page morphological detection...")

            figure_boxes = figure_extraction.find_figure_candidates(self.binarized_inv)

            for i, box in enumerate(figure_boxes):
                box_area = (box[2] - box[0]) * (box[3] - box[1])
                if box_area == 0:
                    continue

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
                if roi_h == 0 or roi_w == 0:
                    continue

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

                figure_counter += 1
                self.elements.append({
                    "id": f"figure_{figure_counter}",
                    "type": "Figure",
                    "bounding_box": box,
                    "kind": kind,
                    "caption": caption,
                    "meta": {"source": "heuristic"}
                })

        figures_in_elements = [el for el in self.elements if el['type'] == 'Figure']
        if figures_in_elements:
            description_generator = self._get_description_generator()

            if description_generator and description_generator.pipe:
                print("    - Generating natural language descriptions for figures...")
                for fig_element in figures_in_elements:
                    kind = fig_element.get("kind", "figure")
                    caption = fig_element.get("caption", "")
                    description = description_generator.generate_for_figure(kind, caption)
                    fig_element['description'] = description

        # Extract chart data if enabled
        if CHART_EXTRACTOR_AVAILABLE and config.EXTRACT_CHART_DATA:
            chart_types = ['bar_chart', 'line_chart', 'pie_chart', 'scatter_plot']
            for fig_element in figures_in_elements:
                kind = fig_element.get("kind", "")
                if kind in chart_types:
                    box = fig_element.get("bounding_box", [])
                    if len(box) == 4:
                        x1, y1, x2, y2 = map(int, box)
                        roi = self.bgr_image[y1:y2, x1:x2]
                        if roi.size > 0:
                            print(f"    - Extracting data from {kind}...")
                            chart_data = chart_extractor.extract_chart_data(roi, kind)
                            if chart_data.get('success', False):
                                fig_element['chart_data'] = chart_data
                                print(f"      Extracted {chart_data.get('num_bars', chart_data.get('num_points', chart_data.get('num_segments', 0)))} data points")

    def _extract_fields(self):
        """Finds key-value pairs (form fields) using hybrid rule-based + LLM approach."""
        print("  [7/8] Extracting key-value fields...")
        text_blocks = [el for el in self.elements if el['type'] == 'TextBlock']

        # Try hybrid extraction if description generator is available
        description_generator = self._get_description_generator()

        if description_generator and description_generator.pipe:
            self.metadata['fields'] = field_extraction.find_fields_hybrid(
                text_blocks,
                description_generator
            )
        else:
            # Fall back to rule-based extraction only
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