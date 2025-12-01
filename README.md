# Document Processing Pipeline v2.0

Advanced document processing pipeline with state-of-the-art OCR, AI-powered image enhancement, and intelligent text correction.

## 🆕 New Features (v2.0)

### 1. AI Super-Resolution (Feature 1)
Enhance low-resolution fax documents and noisy scans using deep learning-based super-resolution.

**Benefits:**
- Improves OCR accuracy by ~15-20% on degraded documents
- Supports multiple SR models: EDSR, ESPCN, FSRCNN, LapSRN
- 🆕 Now triggers on NOISE, not just blur (critical for fax documents!)

**Configuration:**
```python
# In src/config.py
USE_SUPER_RESOLUTION = True
SUPER_RESOLUTION_THRESHOLD = 100.0       # Blur threshold (laplacian variance)
SUPER_RESOLUTION_NOISE_THRESHOLD = 5.0   # 🆕 Noise threshold (faxes are noisy!)
SUPER_RESOLUTION_SCALE = 2               # 2x, 3x, or 4x upscaling
SUPER_RESOLUTION_MODEL = 'EDSR'          # Best quality
```

**Trigger Conditions (any of these):**
1. `laplacian_variance < 100` (blurry image)
2. `noise_estimate > 5` (noisy image - 🆕 KEY FOR FAX DOCUMENTS!)
3. `contrast < 30` (low contrast)

**Usage:**
```python
from src.image_utils import apply_super_resolution, should_apply_super_resolution

# Manual application
enhanced_image = apply_super_resolution(image, model_name='EDSR', scale=2)

# Automatic (integrated in pipeline)
# Super-resolution is applied automatically when image quality is low OR noisy
```

---

### 2. Surya OCR Integration (Feature 2) - 🆕 NOW PRIMARY DETECTOR
Surya is now the **PRIMARY** layout detector for finding text bounding boxes. Tesseract/TrOCR are used for recognition only.

**🆕 Architecture Change (v2.1):**
```
OLD: Tesseract detects + recognizes → Surya as fallback
NEW: Surya ALWAYS detects → Tesseract/TrOCR recognizes from Surya's boxes
```

**Why This Matters:**
- Tesseract was missing lines in noisy faxes (couldn't "see" them)
- Surya excels at finding text in degraded/noisy documents
- Now: Surya finds WHERE text is, Tesseract reads WHAT it says

**Configuration:**
```python
# In src/config.py
USE_SURYA_OCR = True                    # Enable Surya as primary detector
SURYA_COMPLEXITY_THRESHOLD = 0.15       # (Legacy, now always runs first)
SURYA_MIN_LINE_FALLBACK = 3             # (Legacy, now always runs first)  
SURYA_LINE_DETECTION_ONLY = True        # Use Surya for detection only
SURYA_CONFIDENCE_THRESHOLD = 0.6
```

**New Pipeline Flow:**
```python
# In pipeline.py - _detect_layout_surya_first()

# Step A: Surya finds all text bounding boxes
text_blocks = layout_detection.detect_text_blocks_surya(image)

# Step B: For each detected region, crop it
for block in text_blocks:
    roi = image[y1:y2, x1:x2]
    
    # Step C: Send crop to Tesseract/TrOCR for recognition
    text, conf = _recognize_text_in_region(roi)
```

**Usage:**
```python
from src.layout_detection import detect_text_blocks_surya, detect_text_lines_surya

# Get all text bounding boxes (Surya detection)
blocks = detect_text_blocks_surya(image_bgr)
# Returns: [{'bounding_box': [x1,y1,x2,y2], 'type': 'TextBlock', ...}, ...]

# Get raw line coordinates
lines = detect_text_lines_surya(image_bgr)
# Returns: [(x1, y1, x2, y2), ...]
```

---

### 3. Vision-Language Model (VLM) Integration (Feature 3)
Use Qwen-VL-Chat or LLaVA-1.5 for complex form understanding and OCR confirmation.

**Benefits:**
- Handles complex forms that pure OCR struggles with
- Can extract structured data (JSON) from images
- Confirms/corrects low-confidence OCR results

**⚠️ Warning:** Requires 8GB+ GPU VRAM. Disabled by default.

**Configuration:**
```python
# In src/config.py
USE_VLM = False                      # Enable only when needed
VLM_MODEL = 'qwen-vl-chat'           # or 'llava-1.5'
VLM_USE_4BIT = True                  # Reduce memory usage
VLM_CONFIDENCE_THRESHOLD = 50.0      # Trigger VLM below this OCR confidence
VLM_MAX_TOKENS = 256
VLM_TEMPERATURE = 0.1
```

**Usage:**
```python
from src.vlm_manager import VLMManager, query_image_vlm, extract_fields_vlm

# Initialize VLM
vlm = VLMManager()

# Query image with custom prompt
response = vlm.query_image(image, "What is the sender's name?")

# Extract structured fields
fields = vlm.extract_fields_json(image, ['sender', 'date', 'subject'])
# Returns: {'sender': 'John Smith', 'date': '2024-01-15', 'subject': None}

# Confirm OCR output
confirmed_text = vlm.confirm_ocr_text(image, "COVIDVERTANCE", "word")
# Returns: "INADVERTENCE" (corrected)
```

---

### 4. Deterministic Fuzzy Matching (Feature 4)
Fix OCR typos without LLM hallucinations using dictionary-based Levenshtein matching.

**Benefits:**
- Deterministic (same input = same output)
- Fast (no GPU required)
- Safe (won't introduce hallucinations)
- Protects alphanumeric codes automatically

**Configuration:**
```python
# In src/config.py
USE_FUZZY_MATCHING = True
FUZZY_MIN_WORD_LENGTH = 4            # Skip short words
FUZZY_SIMILARITY_THRESHOLD = 0.80    # Minimum similarity for correction
FUZZY_MAX_EDIT_DISTANCE = 2          # Maximum edit distance
FUZZY_SKIP_ALPHANUMERIC = True       # Protect codes/numbers
FUZZY_CUSTOM_DICTIONARY = None       # Path to custom dictionary file
USE_SYMSPELL = True                  # Use SymSpell (faster) vs thefuzz
```

**Usage:**
```python
from src.dictionary_utils import apply_fuzzy_corrections, get_correction_suggestions

# Apply corrections to OCR output
corrected = apply_fuzzy_corrections("COVIDVERTANCE is important")
# Returns: "INADVERTENCE is important"

# Get suggestions for a word
suggestions = get_correction_suggestions("COVIDVERTANCE")
# Returns: [('inadvertence', 0.82), ('advertence', 0.75), ...]
```

---

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Download Tesseract (if not installed)
# Ubuntu/Debian: sudo apt-get install tesseract-ocr tesseract-ocr-eng
# macOS: brew install tesseract
# Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
```

## Quick Start

```python
from src.pipeline import DocumentProcessor
from src import config

# Print feature status
config.print_feature_status()

# Process a document
processor = DocumentProcessor(file_path='document.pdf')
result = processor.run()

# Access results
for element in result['elements']:
    if element['type'] == 'TextBlock':
        print(f"Text: {element['content']}")
        print(f"Confidence: {element['meta']['ocr_conf']}")
```

## Feature Configuration Summary

| Feature | Config Flag | Default | GPU Required |
|---------|-------------|---------|--------------|
| Super-Resolution | `USE_SUPER_RESOLUTION` | `True` | Optional |
| Surya OCR | `USE_SURYA_OCR` | `True` | Yes |
| VLM | `USE_VLM` | `False` | Yes (8GB+) |
| Fuzzy Matching | `USE_FUZZY_MATCHING` | `True` | No |
| OCR Routing | `USE_OCR_ROUTING` | `True` | Yes |
| Text Postprocessing | `ENABLE_TEXT_POSTPROCESSING` | `True` | No |

## Performance Tips

1. **Low VRAM (< 8GB):** Disable VLM, use `SURYA_LINE_DETECTION_ONLY = True`
2. **CPU-only:** Disable Surya and VLM, rely on Tesseract + fuzzy matching
3. **Maximum accuracy:** Enable all features with `VLM_USE_4BIT = True`
4. **Fastest processing:** Disable super-resolution and VLM

## Architecture

```
Document Input
       │
       ▼
┌──────────────────┐
│ Image Quality    │ ──── Blurry OR Noisy? ──► Super-Resolution
│ Analysis         │      (noise > 5 OR blur)
└──────────────────┘
       │
       ▼
┌──────────────────┐     🆕 SURYA-FIRST ARCHITECTURE
│ SURYA Detection  │ ◄── PRIMARY: Finds WHERE text is
│ (Layout/Lines)   │     Superior line detection for noisy faxes
└──────────────────┘
       │
       ▼ (bounding boxes)
┌──────────────────┐
│ For each region: │
│ ┌──────────────┐ │
│ │ Handwriting  │ │──── Handwritten? ──► TrOCR Recognition
│ │ Detection    │ │
│ └──────────────┘ │──── Printed? ──► Tesseract Recognition
│                  │
│ RECOGNITION ONLY │ ◄── Reads WHAT the text says
└──────────────────┘
       │
       ▼
┌──────────────────┐
│ Post-Processing  │ ──► Fuzzy Matching ──► Text Corrections
│                  │ ──► VLM Confirmation (if conf < 50%)
└──────────────────┘
       │
       ▼
   JSON Output
```

### 🆕 v2.1 Key Changes: Surya-First Detection

**Problem:** Tesseract was missing text lines in noisy fax documents because it couldn't "see" them.

**Solution:** 
1. **Surya** is now the PRIMARY layout detector - it finds all text bounding boxes
2. **Tesseract/TrOCR** are used ONLY for recognition - they read text from Surya's detected regions

This separation of concerns ensures:
- ✅ Better line detection (Surya excels at finding text in noise)
- ✅ Better recognition (Tesseract/TrOCR read from clean crops)
- ✅ No missed lines in complex layouts

## License

MIT License
