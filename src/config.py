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

# Denoising 'h' parameter
DENOISING_H = 7

# Adaptive thresholding parameters
ADAPTIVE_THRESH_BLOCK_SIZE = 35
ADAPTIVE_THRESH_C = 11

# Hough Lines threshold for deskew (skew correction)
DESKEW_HOUGH_THRESHOLD = 200

# =============================================================================
#  LAYOUT ANALYSIS SETTINGS
# =============================================================================
# Minimum area to be considered as a text block (square pixels)
BLOCK_MIN_AREA = 500

# Morphological kernel size for merging text lines
MORPH_KERNEL_SIZE = (25, 3)

# Tolerance for grouping columns in reading order (pixels)
COLUMN_REORDER_TOLERANCE = 80

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

# Pixel tolerance for merging nearby lines
TABLE_LINE_MERGE_TOLERANCE = 5

# Minimum width/height to be considered as a cell
TABLE_MIN_CELL_WH = 14

# =============================================================================
#  FIGURE EXTRACTION SETTINGS
# =============================================================================
# Maximum vertical distance to scan for a figure caption (pixels)
FIGURE_CAPTION_V_TOLERANCE = 150