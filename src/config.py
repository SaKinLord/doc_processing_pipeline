# src/config.py

import os
from pathlib import Path

# =============================================================================
#  BASIC FILE PATHS
#  Dynamically finds the project's main directory.
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figure_crops"

# =============================================================================
#  IMAGE PREPROCESSING SETTINGS
# =============================================================================
# DPI value to use when converting PDF to image
PDF_TO_IMAGE_DPI = 300

# Denoising 'h' parameter (Denoising strength)
# CHANGED: Reduced to 5 to preserve faint handwriting strokes
DENOISING_H = 5 

# Adaptive thresholding parameters
# CHANGED: Block size increased; cleans shadowy areas better.
ADAPTIVE_THRESH_BLOCK_SIZE = 45 
ADAPTIVE_THRESH_C = 15

# Hough Lines threshold for deskew (skew correction)
DESKEW_HOUGH_THRESHOLD = 200

# =============================================================================
#  🆕 AI SUPER-RESOLUTION SETTINGS (Feature 1)
# =============================================================================
# Enable AI-based super-resolution for low-quality images
# This can improve OCR accuracy by ~15-20% on degraded documents
USE_SUPER_RESOLUTION = True

# Laplacian variance threshold below which super-resolution is applied
# Lower values indicate blurrier images that benefit from upscaling
SUPER_RESOLUTION_THRESHOLD = 100.0

# 🆕 Noise threshold above which super-resolution is triggered
# Fax documents are typically noisy (noise_estimate > 5), not just blurry!
# This ensures noisy faxes get enhanced even if they're not technically "blurry"
SUPER_RESOLUTION_NOISE_THRESHOLD = 5.0

# Super-resolution scale factor (2x, 3x, or 4x)
# Higher values = more detail but slower processing
SUPER_RESOLUTION_SCALE = 2

# Super-resolution model to use: 'EDSR', 'ESPCN', 'FSRCNN', or 'LapSRN'
# EDSR: Best quality but slowest
# ESPCN: Good balance of speed and quality
# FSRCNN: Fastest but lower quality
# LapSRN: Good for text documents
SUPER_RESOLUTION_MODEL = 'EDSR'

# =============================================================================
#  LAYOUT ANALYSIS SETTINGS
# =============================================================================
# Enable LayoutParser for layout detection (if False, falls back to Tesseract)
# Note: LayoutParser with PaddleDetection has known limitations with text detection.
# Tesseract often provides better text detection results. Set to False if experiencing
# poor text detection or class collapse issues.
USE_LAYOUTPARSER = False  # Recommended: Keep disabled for better text detection

# Minimum area to be considered as a text block (square pixels)
BLOCK_MIN_AREA = 500

# Morphological kernel size for merging text lines
MORPH_KERNEL_SIZE = (25, 3)

# Tolerance for grouping columns in reading order (pixels)
COLUMN_REORDER_TOLERANCE = 80

# Use graph-based topological sorting for reading order (handles complex layouts)
USE_GRAPH_BASED_READING_ORDER = True

# =============================================================================
#  OCR SETTINGS
# =============================================================================
# Default language(s) for Tesseract
DEFAULT_OCR_LANG = "eng+tur"

# Default Page Segmentation Mode (PSM) for Tesseract
DEFAULT_PSM = 6

# Confidence score threshold below which OCR result is considered weak
# Alternatives like PaddleOCR can be tried if below this threshold.
LOW_CONFIDENCE_THRESHOLD = 70.0

# NEW: Confidence threshold for language detection.
# If the probability of the detected language is below this value and the document
# already has a determined language, that language will be used.
LANG_DETECT_CONFIDENCE_THRESHOLD = 0.98

# =============================================================================
#  🆕 SURYA OCR SETTINGS (Feature 2)
# =============================================================================
# Enable Surya OCR for better text line detection and recall
# Surya excels at detecting text in complex layouts where Tesseract fails
USE_SURYA_OCR = True

# Prioritize Surya for complex documents based on text density
# If text density is above this threshold, document is considered complex
SURYA_COMPLEXITY_THRESHOLD = 0.15

# Minimum line count from Tesseract before considering Surya as fallback
# If Tesseract detects fewer lines, Surya will be tried
SURYA_MIN_LINE_FALLBACK = 3

# Use Surya for line detection only (text recognition via Tesseract)
# Set to False to use Surya's full OCR (slower but more accurate)
SURYA_LINE_DETECTION_ONLY = True

# Surya confidence threshold - below this, try other engines
SURYA_CONFIDENCE_THRESHOLD = 0.6

# =============================================================================
#  🆕 VISION-LANGUAGE MODEL (VLM) SETTINGS (Feature 3)
# =============================================================================
# Enable VLM for complex form understanding and confirmation
# WARNING: This feature requires significant GPU memory (8GB+ VRAM)
USE_VLM = False  # Disabled by default due to high GPU requirements

# VLM model to use: 'qwen-vl-chat' or 'llava-1.5'
# qwen-vl-chat: Better for structured extraction, supports Chinese
# llava-1.5: Better general understanding, English-focused
VLM_MODEL = 'qwen-vl-chat'

# Use VLM in 4-bit quantization mode to reduce memory usage
# Recommended for GPUs with less than 16GB VRAM
VLM_USE_4BIT = True

# OCR confidence threshold below which VLM confirmation is triggered
# Only applies when USE_VLM is True
VLM_CONFIDENCE_THRESHOLD = 50.0

# Maximum tokens for VLM response
VLM_MAX_TOKENS = 256

# VLM temperature (0.0 = deterministic, higher = more creative)
VLM_TEMPERATURE = 0.1

# =============================================================================
#  🆕 DICTIONARY-BASED FUZZY MATCHING SETTINGS (Feature 4)
# =============================================================================
# Enable deterministic fuzzy matching for OCR typo correction
# This uses Levenshtein distance to fix common OCR errors without LLM hallucinations
USE_FUZZY_MATCHING = True

# Minimum word length to apply fuzzy matching (shorter words have higher false positive risk)
FUZZY_MIN_WORD_LENGTH = 4

# Levenshtein similarity threshold (0.0-1.0) for accepting a correction
# Higher = stricter matching, fewer corrections
FUZZY_SIMILARITY_THRESHOLD = 0.80

# Maximum edit distance allowed for fuzzy matching
FUZZY_MAX_EDIT_DISTANCE = 2

# Skip fuzzy matching for words containing digits or special patterns
# This protects alphanumeric codes, serial numbers, dates, etc.
FUZZY_SKIP_ALPHANUMERIC = True

# Use domain-specific dictionary in addition to standard English
# Set to a path string to load a custom dictionary file (one word per line)
FUZZY_CUSTOM_DICTIONARY = None

# Enable SymSpell for faster fuzzy matching (recommended for production)
# Falls back to thefuzz if SymSpell is not available
USE_SYMSPELL = True

# =============================================================================
#  TABLE EXTRACTION SETTINGS
# =============================================================================
# Hough Lines parameters for lined tables
TABLE_HOUGH_MIN_LINE_LEN = 80
TABLE_HOUGH_MAX_LINE_GAP = 10


def get_adaptive_line_params(page_width, page_height):
    """
    Calculate adaptive Hough line parameters based on page dimensions.

    Args:
        page_width: Width of the page in pixels
        page_height: Height of the page in pixels

    Returns:
        dict: Adaptive line detection parameters
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

# Pixel tolerance for merging nearby lines
TABLE_LINE_MERGE_TOLERANCE = 5

# Minimum width/height to be considered as a cell
TABLE_MIN_CELL_WH = 14

# Enable advanced table structure analysis (merged cells, HTML/Markdown export)
USE_ADVANCED_TABLE_STRUCTURE = True

# =============================================================================
#  FIGURE EXTRACTION SETTINGS
# =============================================================================
# Maximum vertical distance to scan for a figure caption (pixels)
FIGURE_CAPTION_V_TOLERANCE = 150

# Enable chart data extraction from figures (bar charts, line charts, etc.)
EXTRACT_CHART_DATA = True

# =============================================================================
#  OCR ROUTING SETTINGS
# =============================================================================
# Enable OCR routing based on handwriting detection
# When True, uses pre-classification to route to optimal OCR engine
# When False, uses the fallback chain approach
USE_OCR_ROUTING = True

# Minimum confidence for handwriting detection to trigger direct routing
# Below this threshold, falls back to smart OCR (fallback chain)
HANDWRITING_DETECTION_THRESHOLD = 0.7

# =============================================================================
#  TEXT POST-PROCESSING SETTINGS
# =============================================================================
# Enable automatic text corrections for OCR output
# When True, applies corrections for common OCR errors and spacing issues
# When False, returns raw OCR text without post-processing
ENABLE_TEXT_POSTPROCESSING = True


# =============================================================================
#  FEATURE STATUS SUMMARY (for logging)
# =============================================================================
def get_feature_status():
    """Returns a dictionary summarizing enabled/disabled advanced features."""
    return {
        'super_resolution': USE_SUPER_RESOLUTION,
        'surya_ocr': USE_SURYA_OCR,
        'vlm_confirmation': USE_VLM,
        'fuzzy_matching': USE_FUZZY_MATCHING,
        'ocr_routing': USE_OCR_ROUTING,
        'text_postprocessing': ENABLE_TEXT_POSTPROCESSING,
        'layoutparser': USE_LAYOUTPARSER,
        'advanced_table_structure': USE_ADVANCED_TABLE_STRUCTURE,
    }


def print_feature_status():
    """Prints the status of all advanced features."""
    status = get_feature_status()
    print("\n📊 Advanced Feature Status:")
    print("-" * 40)
    for feature, enabled in status.items():
        icon = "✅" if enabled else "❌"
        print(f"  {icon} {feature.replace('_', ' ').title()}: {'Enabled' if enabled else 'Disabled'}")
    print("-" * 40)
