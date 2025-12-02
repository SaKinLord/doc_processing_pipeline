# src/config.py
# 🆕 OPTIMIZED VERSION for T4 GPU - Reduces processing time by ~50%

import os
from pathlib import Path

# =============================================================================
#  BASIC FILE PATHS
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figure_crops"

# =============================================================================
#  IMAGE PREPROCESSING SETTINGS
# =============================================================================
PDF_TO_IMAGE_DPI = 300
DENOISING_H = 5 
ADAPTIVE_THRESH_BLOCK_SIZE = 45 
ADAPTIVE_THRESH_C = 15
DESKEW_HOUGH_THRESHOLD = 200

# =============================================================================
#  🆕 AI SUPER-RESOLUTION SETTINGS (OPTIMIZED)
# =============================================================================
# CHANGE: Set to False for fastest processing, or True with FSRCNN for balance
USE_SUPER_RESOLUTION = True

# CHANGE: Increased from 100.0 - fewer blur triggers
SUPER_RESOLUTION_THRESHOLD = 80.0

# CHANGE: Increased from 5.0 to 8.0 - triggers SR less often on faxes
# Set to 10.0 or higher if you want even fewer SR triggers
SUPER_RESOLUTION_NOISE_THRESHOLD = 8.0

SUPER_RESOLUTION_SCALE = 2

# CHANGE: FSRCNN is ~3x faster than EDSR with acceptable quality
# Options: 'EDSR' (best quality), 'FSRCNN' (fastest), 'ESPCN' (balanced)
SUPER_RESOLUTION_MODEL = 'FSRCNN'

# =============================================================================
#  LAYOUT ANALYSIS SETTINGS
# =============================================================================
USE_LAYOUTPARSER = True
BLOCK_MIN_AREA = 500
MORPH_KERNEL_SIZE = (25, 3)
COLUMN_REORDER_TOLERANCE = 80
USE_GRAPH_BASED_READING_ORDER = True

# =============================================================================
#  OCR SETTINGS
# =============================================================================
DEFAULT_OCR_LANG = "eng+tur"
DEFAULT_PSM = 6
LOW_CONFIDENCE_THRESHOLD = 70.0
LANG_DETECT_CONFIDENCE_THRESHOLD = 0.98

# =============================================================================
#  🆕 SURYA OCR SETTINGS
# =============================================================================
USE_SURYA_OCR = True
SURYA_COMPLEXITY_THRESHOLD = 0.15
SURYA_MIN_LINE_FALLBACK = 3
SURYA_LINE_DETECTION_ONLY = True
SURYA_CONFIDENCE_THRESHOLD = 0.6

# =============================================================================
#  🆕 VISION-LANGUAGE MODEL (VLM) SETTINGS (OPTIMIZED)
# =============================================================================
# CHANGE: Disabled by default for faster processing
# VLM adds ~30-60 seconds per page on T4
# Enable if form field extraction is critical
USE_VLM = True

VLM_MODEL = 'qwen-vl-chat'

# CHANGE: Use float16 compute dtype for T4 compatibility
# Even if VLM is enabled, this prevents bfloat16 errors
VLM_USE_4BIT = False  # Disabled because triton/bitsandbytes has issues on T4

VLM_CONFIDENCE_THRESHOLD = 40.0  # Only use VLM for very low confidence OCR
VLM_MAX_TOKENS = 256
VLM_TEMPERATURE = 0.1

# =============================================================================
#  🆕 DICTIONARY-BASED FUZZY MATCHING SETTINGS
# =============================================================================
USE_FUZZY_MATCHING = True
FUZZY_MIN_WORD_LENGTH = 4
FUZZY_SIMILARITY_THRESHOLD = 0.80
FUZZY_MAX_EDIT_DISTANCE = 2
FUZZY_SKIP_ALPHANUMERIC = True
FUZZY_CUSTOM_DICTIONARY = None
USE_SYMSPELL = True

# =============================================================================
#  TABLE EXTRACTION SETTINGS
# =============================================================================
TABLE_HOUGH_MIN_LINE_LEN = 80
TABLE_HOUGH_MAX_LINE_GAP = 10


def get_adaptive_line_params(page_width, page_height):
    """Calculate adaptive Hough line parameters based on page dimensions."""
    min_dimension = min(page_width, page_height)
    min_line_length = max(40, int(min_dimension * 0.05))
    table_min_line_length = max(80, int(page_width * 0.10))
    max_line_gap = max(5, int(min_dimension * 0.01))

    return {
        'min_line_length': min_line_length,
        'table_min_line_length': table_min_line_length,
        'max_line_gap': max_line_gap,
    }


TABLE_LINE_MERGE_TOLERANCE = 5
TABLE_MIN_CELL_WH = 14
USE_ADVANCED_TABLE_STRUCTURE = True

# =============================================================================
#  FIGURE EXTRACTION SETTINGS
# =============================================================================
FIGURE_CAPTION_V_TOLERANCE = 150
EXTRACT_CHART_DATA = True

# =============================================================================
#  OCR ROUTING SETTINGS
# =============================================================================
USE_OCR_ROUTING = True
HANDWRITING_DETECTION_THRESHOLD = 0.7

# =============================================================================
#  TEXT POST-PROCESSING SETTINGS
# =============================================================================
ENABLE_TEXT_POSTPROCESSING = True


# =============================================================================
#  FEATURE STATUS SUMMARY
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


# =============================================================================
#  🆕 PERFORMANCE TUNING OPTIONS
# =============================================================================
# These settings help balance speed vs quality on T4 GPU

# Maximum number of text regions to process per page
# Set to None for unlimited, or a number like 100 to cap processing
MAX_TEXT_REGIONS_PER_PAGE = None

# Minimum OCR confidence to trigger LLM correction
# Higher = fewer LLM calls = faster processing
LLM_CORRECTION_MIN_CONFIDENCE = 30  # Only correct if conf > 30
LLM_CORRECTION_MAX_CONFIDENCE = 60  # Only correct if conf < 60 (was 85)

# Skip figure classification for small regions (faster)
MIN_FIGURE_AREA_FOR_CLASSIFICATION = 5000  # Square pixels

# Enable/disable description generator LLM corrections
USE_LLM_OCR_CORRECTIONS = True
