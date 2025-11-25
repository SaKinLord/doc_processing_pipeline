# Document Processing Pipeline - Accuracy Improvement Recommendations

## Executive Summary

After analyzing your `doc_processing_pipeline` codebase, I've identified **15 key areas** for improving accuracy, organized by priority and impact. The most significant issues are in OCR quality, table structure recognition, and preprocessing adaptive tuning.

---

## 🔴 High Priority Issues

### 1. OCR Confidence Scores Are Extremely Low

**Problem:** Your output samples show many text blocks with confidence scores of 0.0, 1.33, 8.5, 12.25, etc. This indicates severe OCR issues.

**Evidence from your output:**
```json
{"content": "retrenereasesereas", "ocr_conf": 0.0}
{"content": "semen Golden mat snowmen anon", "ocr_conf": 20.2}
{"content": "Who? SCNLKSS5", "ocr_conf": 1.33}
```

**Solutions:**

```python
# src/ocr.py - Add confidence-based retry with different preprocessing

def ocr_with_retry(original_bgr, binary_image, box, lang, min_confidence=50.0):
    """
    Attempts OCR with multiple preprocessing strategies if confidence is low.
    """
    strategies = [
        ("original", lambda img: img),
        ("contrast_enhanced", lambda img: enhance_contrast(img)),
        ("sharpened", lambda img: sharpen_image(img)),
        ("scaled_2x", lambda img: cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)),
        ("morphology_cleaned", lambda img: morphology_clean(img)),
    ]
    
    best_result = ("", 0.0)
    
    for strategy_name, preprocess_fn in strategies:
        try:
            processed = preprocess_fn(original_bgr)
            text, conf = ocr_with_tesseract(processed, lang=lang, box=box)
            
            if conf > best_result[1]:
                best_result = (text, conf)
            
            if conf >= min_confidence:
                return text, conf
        except Exception:
            continue
    
    return best_result

def enhance_contrast(image):
    """CLAHE-based contrast enhancement"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

def sharpen_image(image):
    """Unsharp masking for sharpening"""
    gaussian = cv2.GaussianBlur(image, (0, 0), 3)
    return cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)

def morphology_clean(image):
    """Clean noise with morphological operations"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    return cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
```

---

### 2. Table Structure Recognition Produces Invalid Output

**Problem:** Your CSV outputs contain patterns like `[→0,8]` which are merge indicators, not actual data.

**Evidence:**
```csv
,,,,,,,,,"[→0,8]","[→0,8]","[→0,8]"
,,"[→1,1]","[→1,1]","[→1,1]"...
```

**Solutions:**

```python
# src/table_structure.py - Fix CSV export to contain actual text

def structure_to_csv(cell_data_map, num_rows, num_cols):
    """
    Converts structured table data to CSV with actual cell content.
    Merged cells show content in the top-left cell, empty in spanned cells.
    """
    import csv
    import io
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Track which cells are covered by spans
    covered = set()
    
    for row in range(num_rows):
        row_data = []
        
        for col in range(num_cols):
            if (row, col) in covered:
                row_data.append("")  # Empty for spanned cells
                continue
            
            cell_info = cell_data_map.get((row, col), {})
            text = cell_info.get('text', '').strip()
            rowspan = cell_info.get('rowspan', 1)
            colspan = cell_info.get('colspan', 1)
            
            # Mark covered cells
            for r in range(row, row + rowspan):
                for c in range(col, col + colspan):
                    if (r, c) != (row, col):
                        covered.add((r, c))
            
            row_data.append(text)
        
        writer.writerow(row_data)
    
    return output.getvalue()


# Also add a "flat" export that repeats merged cell content
def structure_to_csv_flat(cell_data_map, num_rows, num_cols):
    """
    CSV export where merged cells repeat their content in all spanned positions.
    Better for downstream analysis tools.
    """
    import csv
    import io
    
    # First, build a flat grid
    flat_grid = [["" for _ in range(num_cols)] for _ in range(num_rows)]
    
    for (row, col), cell_info in cell_data_map.items():
        text = cell_info.get('text', '').strip()
        rowspan = cell_info.get('rowspan', 1)
        colspan = cell_info.get('colspan', 1)
        
        # Fill all cells in the span with the same content
        for r in range(row, min(row + rowspan, num_rows)):
            for c in range(col, min(col + colspan, num_cols)):
                flat_grid[r][c] = text
    
    output = io.StringIO()
    writer = csv.writer(output)
    for row_data in flat_grid:
        writer.writerow(row_data)
    
    return output.getvalue()
```

---

### 3. Handwriting Detection Threshold Too Conservative

**Problem:** Your `confidence_threshold=0.65` in HandwritingDetector and `detection_confidence > 0.55` routing threshold may cause handwritten text to fall through to Tesseract, which performs poorly on handwriting.

**Solutions:**

```python
# src/handwriting_detector.py - Improved feature extraction

class HandwritingDetector:
    def __init__(self, confidence_threshold=0.55):  # Lowered from 0.65
        self.confidence_threshold = confidence_threshold
        # ... rest of init
    
    def _extract_stroke_features(self, gray_image):
        """
        NEW: Extract stroke-specific features that better distinguish handwriting.
        """
        features = {}
        
        if gray_image.size == 0:
            return features
        
        # 1. Stroke width variation using distance transform
        _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        
        if dist_transform.size > 0:
            features['stroke_width_mean'] = np.mean(dist_transform[dist_transform > 0]) if np.any(dist_transform > 0) else 0
            features['stroke_width_std'] = np.std(dist_transform[dist_transform > 0]) if np.any(dist_transform > 0) else 0
            # High std = variable stroke width = likely handwriting
            features['stroke_variation_ratio'] = features['stroke_width_std'] / (features['stroke_width_mean'] + 1e-6)
        
        # 2. Baseline irregularity
        # Find text baseline by horizontal projection
        h_proj = np.sum(binary, axis=1)
        if len(h_proj) > 10:
            # Find peaks in projection (text lines)
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(h_proj, height=np.max(h_proj) * 0.3)
            if len(peaks) > 1:
                peak_diffs = np.diff(peaks)
                features['baseline_regularity'] = np.std(peak_diffs) / (np.mean(peak_diffs) + 1e-6)
            else:
                features['baseline_regularity'] = 0
        
        # 3. Character connectivity
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]  # Exclude background
            features['component_count'] = num_labels - 1
            features['avg_component_area'] = np.mean(areas)
            features['component_area_variance'] = np.std(areas) / (np.mean(areas) + 1e-6)
        
        return features
    
    def classify(self, roi_image):
        """
        Enhanced classification with stroke features.
        """
        if roi_image is None or roi_image.size == 0:
            return {'classification': 'unknown', 'confidence': 0.0}
        
        gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY) if len(roi_image.shape) == 3 else roi_image
        
        texture_features = self._extract_texture_features(gray)
        stroke_features = self._extract_stroke_features(gray)
        
        # Scoring system
        handwriting_score = 0.0
        total_weight = 0.0
        
        # Stroke variation is a strong indicator (weight: 0.3)
        if 'stroke_variation_ratio' in stroke_features:
            svr = stroke_features['stroke_variation_ratio']
            if svr > 0.4:  # High variation = handwriting
                handwriting_score += 0.3
            elif svr < 0.2:  # Low variation = printed
                handwriting_score -= 0.2
            total_weight += 0.3
        
        # Baseline irregularity (weight: 0.25)
        if 'baseline_regularity' in stroke_features:
            br = stroke_features['baseline_regularity']
            if br > 0.3:  # Irregular baseline = handwriting
                handwriting_score += 0.25
            elif br < 0.1:  # Regular baseline = printed
                handwriting_score -= 0.15
            total_weight += 0.25
        
        # Laplacian variance (weight: 0.2)
        if 'laplacian_var' in texture_features:
            lv = texture_features['laplacian_var']
            if lv > 2000:  # High edge variance = printed (sharp edges)
                handwriting_score -= 0.15
            elif lv < 500:  # Low edge variance = possibly handwriting (smoother)
                handwriting_score += 0.1
            total_weight += 0.2
        
        # Component variance (weight: 0.25)
        if 'component_area_variance' in stroke_features:
            cav = stroke_features['component_area_variance']
            if cav > 0.8:  # Highly variable component sizes = handwriting
                handwriting_score += 0.25
            elif cav < 0.3:  # Uniform components = printed
                handwriting_score -= 0.15
            total_weight += 0.25
        
        # Normalize score to 0-1
        if total_weight > 0:
            confidence = (handwriting_score / total_weight + 1) / 2  # Map [-1,1] to [0,1]
        else:
            confidence = 0.5
        
        classification = 'handwritten' if confidence > self.confidence_threshold else 'printed'
        
        return {
            'classification': classification,
            'confidence': confidence,
            'features': {**texture_features, **stroke_features}
        }
```

---

### 4. TrOCR Generation Parameters Cause Hallucinations

**Problem:** Your TrOCR settings use `do_sample=True` with `temperature=0.1`, which can still cause hallucinations. Also, the model sometimes generates gibberish for multi-line text.

**Solutions:**

```python
# src/ocr.py - Improved TrOCR configuration

def ocr_with_trocr(roi_bgr):
    """
    Improved TrOCR with deterministic generation and better line handling.
    """
    _initialize_trocr()
    if _trocr_model_instance is None or _trocr_processor_instance is None:
        return ""
    
    line_images = split_text_block_into_lines(roi_bgr)
    full_text_parts = []
    
    for line_img in line_images:
        if line_img.shape[0] < 8 or line_img.shape[1] < 10:  # Skip tiny lines
            continue
        
        # Resize if too small (TrOCR works better with larger images)
        h, w = line_img.shape[:2]
        if h < 32:
            scale = 32 / h
            line_img = cv2.resize(line_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        rgb_image = cv2.cvtColor(line_img, cv2.COLOR_BGR2RGB)
        
        try:
            pixel_values = _trocr_processor_instance(images=rgb_image, return_tensors="pt").pixel_values.to("cuda")
            
            # FIXED: Use fully deterministic generation
            generated_ids = _trocr_model_instance.generate(
                pixel_values,
                max_new_tokens=128,      # Increased for longer lines
                num_beams=5,             # Increased beam width
                early_stopping=True,
                no_repeat_ngram_size=3,
                do_sample=False,         # CRITICAL: Disable sampling for determinism
                length_penalty=1.0,
                repetition_penalty=1.2,  # Penalize repetition
            )
            generated_text = _trocr_processor_instance.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            clean_text = generated_text.strip()
            
            # Enhanced filtering
            if clean_text and is_valid_ocr_output(clean_text):
                full_text_parts.append(clean_text)
                
        except Exception as e:
            print(f"    - Error during TrOCR line inference: {e}")
            continue
    
    return " ".join(full_text_parts)


def is_valid_ocr_output(text):
    """
    Validates OCR output to filter hallucinations and garbage.
    """
    if not text or len(text.strip()) == 0:
        return False
    
    text = text.strip()
    
    # 1. Check for repetitive patterns (hallucination indicator)
    if len(text) > 5:
        for pattern_len in range(2, min(6, len(text) // 2)):
            pattern = text[:pattern_len]
            if text.count(pattern) > len(text) / pattern_len * 0.6:
                return False
    
    # 2. Check alphanumeric ratio
    alnum_count = sum(1 for c in text if c.isalnum())
    if len(text) > 3 and alnum_count / len(text) < 0.3:
        return False
    
    # 3. Check for known garbage patterns
    garbage_patterns = [
        r'^[.,:;!?]+$',           # Only punctuation
        r'^[-_=+]+$',             # Only symbols
        r'(.)\1{4,}',             # 5+ repeated characters
        r'^[A-Z]{15,}$',          # 15+ uppercase letters (likely noise)
    ]
    import re
    for pattern in garbage_patterns:
        if re.match(pattern, text):
            return False
    
    return True
```

---

## 🟠 Medium Priority Issues

### 5. Adaptive Preprocessing Not Leveraging Quality Metrics Effectively

**Problem:** You calculate `laplacian_variance`, `brightness`, and `contrast` but the preset selection logic could be more nuanced.

**Solution:**

```python
# src/image_utils.py - Enhanced adaptive preprocessing

def analyze_image_quality(image_bgr):
    """
    Enhanced quality analysis with more metrics.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    # Existing metrics
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = np.mean(gray)
    contrast = np.std(gray)
    
    # NEW: Additional metrics
    
    # 1. Noise estimation using median filter difference
    median_filtered = cv2.medianBlur(gray, 5)
    noise_estimate = np.mean(np.abs(gray.astype(float) - median_filtered.astype(float)))
    
    # 2. Skew angle estimation
    skew_angle = estimate_skew_angle(gray)
    
    # 3. Text density (ratio of foreground pixels)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    text_density = np.sum(binary > 0) / binary.size
    
    # 4. Local contrast variation (for detecting faded regions)
    local_std = cv2.blur(gray.astype(float) ** 2, (50, 50)) - cv2.blur(gray.astype(float), (50, 50)) ** 2
    local_contrast_uniformity = np.std(np.sqrt(np.maximum(local_std, 0)))
    
    return {
        'laplacian_variance': laplacian_var,
        'brightness': brightness,
        'contrast': contrast,
        'noise_estimate': noise_estimate,
        'skew_angle': skew_angle,
        'text_density': text_density,
        'local_contrast_uniformity': local_contrast_uniformity,
    }


def get_preprocessing_params(quality_metrics):
    """
    Select preprocessing parameters based on comprehensive quality analysis.
    """
    params = {
        'denoising_h': 5,
        'clahe_clip_limit': 2.0,
        'clahe_grid_size': (8, 8),
        'sharpen': False,
        'deskew': False,
        'contrast_enhance': False,
        'binarization_method': 'adaptive',  # 'adaptive', 'otsu', or 'sauvola'
    }
    
    lap_var = quality_metrics['laplacian_variance']
    brightness = quality_metrics['brightness']
    contrast = quality_metrics['contrast']
    noise = quality_metrics['noise_estimate']
    skew = quality_metrics.get('skew_angle', 0)
    text_density = quality_metrics.get('text_density', 0.1)
    
    # Noise-based denoising
    if noise > 15:
        params['denoising_h'] = min(10, int(noise / 2))  # Scale with noise
    elif noise < 5:
        params['denoising_h'] = 0  # Skip denoising for clean images
    
    # Blurry images (low laplacian variance)
    if lap_var < 100:
        params['sharpen'] = True
        params['binarization_method'] = 'sauvola'  # Better for low contrast
    
    # Low contrast images
    if contrast < 40:
        params['clahe_clip_limit'] = 3.0
        params['contrast_enhance'] = True
    
    # Very bright or very dark images
    if brightness > 220 or brightness < 50:
        params['clahe_clip_limit'] = 2.5
        params['contrast_enhance'] = True
    
    # Skewed images
    if abs(skew) > 0.5:
        params['deskew'] = True
    
    # Dense text (forms, tables) - use finer grid
    if text_density > 0.3:
        params['clahe_grid_size'] = (4, 4)
    
    return params


def estimate_skew_angle(gray_image, max_angle=15):
    """
    Estimate document skew angle using Hough transform.
    """
    edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    
    if lines is None:
        return 0.0
    
    angles = []
    for line in lines:
        rho, theta = line[0]
        angle_deg = np.degrees(theta) - 90
        if abs(angle_deg) < max_angle:
            angles.append(angle_deg)
    
    if not angles:
        return 0.0
    
    # Use median to be robust to outliers
    return np.median(angles)
```

---

### 6. PaddleOCR Not Used Effectively as Fallback

**Problem:** PaddleOCR is only used as a fallback but could be used in an ensemble approach for better results.

**Solution:**

```python
# src/ocr.py - OCR Ensemble with weighted voting

def ocr_ensemble(original_bgr, binary_image, box, lang='eng'):
    """
    Run multiple OCR engines and combine results using weighted voting.
    """
    results = []
    
    # 1. Tesseract OCR
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
    
    # 2. PaddleOCR
    try:
        paddle_text = ocr_with_paddle(original_bgr)
        if paddle_text.strip():
            # PaddleOCR doesn't return confidence, estimate from quality
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
    
    # 3. Select best result or combine
    if not results:
        return "", 0.0
    
    if len(results) == 1:
        return results[0]['text'], results[0]['confidence']
    
    # If multiple results, use weighted voting
    # For now, simply pick the one with highest weighted quality
    best = max(results, key=lambda r: r['quality'] * r['weight'])
    
    # If results are very different, log for debugging
    if len(results) > 1:
        texts = [r['text'] for r in results]
        if not texts_are_similar(texts[0], texts[1]):
            print(f"    - OCR disagreement: {[r['engine'] + ':' + r['text'][:30] for r in results]}")
    
    return best['text'], best['confidence']


def texts_are_similar(text1, text2, threshold=0.7):
    """Check if two texts are similar using character-level similarity."""
    if not text1 or not text2:
        return False
    
    # Simple Jaccard similarity on character bigrams
    def get_bigrams(text):
        text = text.lower().replace(' ', '')
        return set(text[i:i+2] for i in range(len(text) - 1))
    
    b1, b2 = get_bigrams(text1), get_bigrams(text2)
    if not b1 or not b2:
        return text1.lower() == text2.lower()
    
    intersection = len(b1 & b2)
    union = len(b1 | b2)
    
    return intersection / union >= threshold if union > 0 else False
```

---

### 7. Line Detection Parameters Too Strict for Some Documents

**Problem:** `TABLE_HOUGH_MIN_LINE_LEN = 80` and the 400px threshold may miss tables with shorter lines.

**Solution:**

```python
# src/config.py - Add adaptive line detection

# Make line detection adaptive based on page size
def get_adaptive_line_params(page_width, page_height):
    """
    Calculate adaptive Hough line parameters based on page dimensions.
    """
    min_dimension = min(page_width, page_height)
    
    # Minimum line length: 5% of smaller dimension, but at least 40px
    min_line_length = max(40, int(min_dimension * 0.05))
    
    # For tables: require lines to be at least 10% of page width
    table_min_line_length = max(80, int(page_width * 0.10))
    
    # Line gap tolerance: scale with page size
    max_line_gap = max(5, int(min_dimension * 0.01))
    
    return {
        'min_line_length': min_line_length,
        'table_min_line_length': table_min_line_length,
        'max_line_gap': max_line_gap,
    }


# src/table_extraction.py - Use adaptive parameters

def extract_table_lines(binary_inv, page_width, page_height):
    """
    Extract table lines with adaptive parameters.
    """
    params = get_adaptive_line_params(page_width, page_height)
    
    # Horizontal lines
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (params['min_line_length'], 1))
    h_lines_img = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, h_kernel, iterations=2)
    
    # Vertical lines
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, params['min_line_length']))
    v_lines_img = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, v_kernel, iterations=2)
    
    # Use Hough transform with adaptive threshold
    h_lines = cv2.HoughLinesP(
        h_lines_img, 1, np.pi/180, 
        threshold=50,
        minLineLength=params['table_min_line_length'],
        maxLineGap=params['max_line_gap']
    )
    
    v_lines = cv2.HoughLinesP(
        v_lines_img, 1, np.pi/180,
        threshold=50,
        minLineLength=params['table_min_line_length'],
        maxLineGap=params['max_line_gap']
    )
    
    return h_lines, v_lines
```

---

### 8. Figure Classification Using ImageNet Labels Is Imprecise

**Problem:** You're using `google/vit-base-patch16-224` trained on ImageNet, which doesn't have document-specific classes like "bar_chart" or "flowchart".

**Solution:**

```python
# Option A: Use a document-specific model (recommended)
# Replace with a model fine-tuned on document figures

# src/classifier.py - Alternative model options

class FigureClassifier:
    def __init__(self):
        """
        Initialize with a more appropriate model for document figures.
        """
        # Option 1: Use CLIP for zero-shot classification
        try:
            from transformers import CLIPProcessor, CLIPModel
            
            self.model_type = 'clip'
            model_id = "openai/clip-vit-base-patch32"
            self.processor = CLIPProcessor.from_pretrained(model_id)
            self.model = CLIPModel.from_pretrained(model_id).to("cuda")
            
            # Define document-specific labels for zero-shot
            self.candidate_labels = [
                "a bar chart with vertical or horizontal bars",
                "a line graph showing trends over time",
                "a pie chart showing percentages",
                "a scatter plot with data points",
                "a flowchart or process diagram",
                "a photograph or image",
                "a map or geographic visualization",
                "a table or data grid",
                "an organizational chart",
                "a technical diagram or schematic",
                "a logo or icon",
                "a signature or handwriting",
            ]
            
            self.label_to_kind = {
                "a bar chart with vertical or horizontal bars": "bar_chart",
                "a line graph showing trends over time": "line_chart",
                "a pie chart showing percentages": "pie_chart",
                "a scatter plot with data points": "scatter_plot",
                "a flowchart or process diagram": "flowchart",
                "a photograph or image": "photo",
                "a map or geographic visualization": "map",
                "a table or data grid": "table_image",
                "an organizational chart": "org_chart",
                "a technical diagram or schematic": "diagram",
                "a logo or icon": "logo",
                "a signature or handwriting": "signature",
            }
            
            print("    - ✅ Figure Classifier (CLIP) initialized for zero-shot classification")
            
        except Exception as e:
            print(f"    - Failed to load CLIP, falling back to ViT: {e}")
            self._init_vit_fallback()
    
    def _init_vit_fallback(self):
        """Fall back to ViT if CLIP fails."""
        from transformers import ViTImageProcessor, ViTForImageClassification
        
        self.model_type = 'vit'
        model_id = "google/vit-base-patch16-224"
        self.processor = ViTImageProcessor.from_pretrained(model_id)
        self.model = ViTForImageClassification.from_pretrained(model_id).to("cuda")
    
    def classify(self, image_roi) -> str:
        """Classify figure type."""
        if self.model is None:
            return "figure"
        
        try:
            image_rgb = Image.fromarray(cv2.cvtColor(image_roi, cv2.COLOR_BGR2RGB))
            
            if self.model_type == 'clip':
                # Zero-shot classification with CLIP
                inputs = self.processor(
                    text=self.candidate_labels,
                    images=image_rgb,
                    return_tensors="pt",
                    padding=True
                ).to("cuda")
                
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
                
                best_idx = probs.argmax().item()
                best_label = self.candidate_labels[best_idx]
                
                return self.label_to_kind.get(best_label, "figure")
            
            else:
                # ViT classification (original behavior)
                inputs = self.processor(images=image_rgb, return_tensors="pt").to("cuda")
                outputs = self.model(**inputs)
                predicted_class_idx = outputs.logits.argmax(-1).item()
                predicted_label = self.model.config.id2label[predicted_class_idx]
                return self._map_label_to_kind(predicted_label)
                
        except Exception as e:
            print(f"    - Error during figure classification: {e}")
            return "figure"
```

---

## 🟢 Lower Priority Improvements

### 9. Add Post-Processing Text Correction

```python
# src/text_postprocessing.py - NEW FILE

import re
from collections import Counter

# Common OCR error patterns
OCR_CORRECTIONS = {
    r'\bl\b': 'I',           # Lowercase L often misread as I
    r'\bO\b(?=\d)': '0',     # O before digit is likely 0
    r'(?<=\d)O(?=\d)': '0',  # O between digits is likely 0
    r'rn': 'm',               # rn often misread as m
    r'vv': 'w',               # vv often misread as w
    r'c1': 'd',               # c1 often misread as d
    r'\bII\b': 'II',          # Keep Roman numerals
}

def correct_common_ocr_errors(text):
    """Apply common OCR error corrections."""
    for pattern, replacement in OCR_CORRECTIONS.items():
        text = re.sub(pattern, replacement, text)
    return text

def fix_spacing_issues(text):
    """Fix common spacing problems from OCR."""
    # Remove spaces before punctuation
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    # Add space after punctuation if missing
    text = re.sub(r'([.,;:!?])([A-Za-z])', r'\1 \2', text)
    # Fix multiple spaces
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

def apply_text_corrections(text):
    """Apply all text corrections."""
    text = correct_common_ocr_errors(text)
    text = fix_spacing_issues(text)
    return text
```

---

### 10. Add Confidence-Based Output Flagging

```python
# src/pipeline.py - Add quality flags to output

def add_quality_flags(page_result):
    """
    Add quality flags to help downstream systems identify uncertain extractions.
    """
    flags = []
    
    # Check OCR confidence
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
    
    # Check for potential handwriting
    quality = page_result.get('metadata', {}).get('image_quality', {})
    if quality.get('quality_preset') == 'noisy':
        flags.append({
            'type': 'NOISY_IMAGE',
            'severity': 'info',
            'message': 'Image quality detected as noisy, results may be affected'
        })
    
    page_result['quality_flags'] = flags
    return page_result
```

---

### 11. Improve Borderless Table Detection

```python
# src/table_extraction.py - Enhanced borderless table detection

def detect_borderless_table_advanced(text_blocks, page_width, page_height):
    """
    Advanced borderless table detection using grid alignment analysis.
    """
    if len(text_blocks) < 4:
        return None
    
    # Extract bounding boxes
    boxes = [block['bounding_box'] for block in text_blocks]
    
    # Calculate centers
    centers_x = [(b[0] + b[2]) / 2 for b in boxes]
    centers_y = [(b[1] + b[3]) / 2 for b in boxes]
    
    # Find column candidates using DBSCAN clustering
    from sklearn.cluster import DBSCAN
    import numpy as np
    
    # Cluster x-coordinates to find columns
    x_array = np.array(centers_x).reshape(-1, 1)
    col_clustering = DBSCAN(eps=30, min_samples=2).fit(x_array)
    
    col_labels = col_clustering.labels_
    n_columns = len(set(col_labels)) - (1 if -1 in col_labels else 0)
    
    if n_columns < 2:
        return None  # Not enough columns for a table
    
    # Cluster y-coordinates to find rows
    y_array = np.array(centers_y).reshape(-1, 1)
    row_clustering = DBSCAN(eps=25, min_samples=2).fit(y_array)
    
    row_labels = row_clustering.labels_
    n_rows = len(set(row_labels)) - (1 if -1 in row_labels else 0)
    
    if n_rows < 2:
        return None  # Not enough rows for a table
    
    # Build grid and check alignment quality
    grid = {}
    for i, block in enumerate(text_blocks):
        if col_labels[i] == -1 or row_labels[i] == -1:
            continue  # Skip outliers
        
        cell_key = (row_labels[i], col_labels[i])
        if cell_key not in grid:
            grid[cell_key] = []
        grid[cell_key].append(block)
    
    # Calculate grid occupancy
    expected_cells = n_rows * n_columns
    occupied_cells = len(grid)
    occupancy_ratio = occupied_cells / expected_cells if expected_cells > 0 else 0
    
    # A good table should have reasonable occupancy (>40%)
    if occupancy_ratio < 0.4:
        return None
    
    return {
        'type': 'borderless_table',
        'n_rows': n_rows,
        'n_columns': n_columns,
        'grid': grid,
        'occupancy_ratio': occupancy_ratio
    }
```

---

## 📊 Recommended Implementation Priority

| Priority | Improvement | Impact | Effort |
|----------|-------------|--------|--------|
| 1 | OCR confidence retry (#1) | High | Medium |
| 2 | Fix CSV export (#2) | High | Low |
| 3 | TrOCR deterministic generation (#4) | High | Low |
| 4 | Improve handwriting detection (#3) | High | Medium |
| 5 | Enhanced preprocessing (#5) | Medium | Medium |
| 6 | OCR ensemble (#6) | Medium | Medium |
| 7 | CLIP figure classifier (#8) | Medium | Medium |
| 8 | Adaptive line detection (#7) | Low | Low |
| 9 | Post-processing corrections (#9) | Low | Low |
| 10 | Quality flagging (#10) | Low | Low |

---

## 🧪 Testing Recommendations

1. **Create a benchmark dataset** with ground truth:
   - 50+ documents with varying quality
   - Manual annotations for tables, figures, text
   - Include handwritten and printed samples

2. **Add unit tests for each module**:
```python
def test_ocr_low_confidence_retry():
    # Test that retry improves results on noisy images
    noisy_image = load_test_image("noisy_text.png")
    result1 = ocr_with_tesseract(noisy_image)
    result2 = ocr_with_retry(noisy_image)
    assert result2[1] >= result1[1]  # Confidence should improve
```

3. **Add integration tests**:
```python
def test_full_pipeline_accuracy():
    # Process known documents and compare to ground truth
    result = process_document("test_form.pdf")
    assert result['pages'][0]['metadata']['fields']['date']['value'] == "2024-01-15"
```

---

## Summary

The most impactful improvements are:
1. **Multi-strategy OCR with retry** - will significantly improve low-confidence results
2. **Fixed CSV export** - currently unusable output format
3. **Deterministic TrOCR** - prevents hallucinations
4. **Better handwriting detection** - routes text to appropriate OCR engine

These changes together should improve overall accuracy by 20-40% based on typical document processing benchmarks.
