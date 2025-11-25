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