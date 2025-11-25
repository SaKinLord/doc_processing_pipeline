# src/text_postprocessing.py

"""
Text Post-Processing Module

Applies corrections for common OCR errors and formatting issues.
"""

import re


# Common OCR error patterns and their corrections
OCR_CORRECTIONS = {
    r'\bl\b': 'I',              # Standalone lowercase L → I
    r'\bO\b(?=\d)': '0',        # O before digit → 0
    r'(?<=\d)O(?=\d)': '0',     # O between digits → 0
    r'rn': 'm',                 # rn → m (common misread)
    r'vv': 'w',                 # vv → w
}


def correct_common_ocr_errors(text):
    """
    Apply common OCR error corrections.

    Args:
        text: Input text string

    Returns:
        str: Corrected text
    """
    for pattern, replacement in OCR_CORRECTIONS.items():
        text = re.sub(pattern, replacement, text)
    return text


def fix_spacing_issues(text):
    """
    Fix common spacing problems from OCR.

    Args:
        text: Input text string

    Returns:
        str: Text with corrected spacing
    """
    # Remove spaces before punctuation
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)

    # Add space after punctuation if missing
    text = re.sub(r'([.,;:!?])([A-Za-z])', r'\1 \2', text)

    # Fix multiple spaces
    text = re.sub(r'\s{2,}', ' ', text)

    return text.strip()


def apply_text_corrections(text):
    """
    Apply all text corrections.

    Args:
        text: Input text string

    Returns:
        str: Fully corrected text
    """
    if not text:
        return text

    text = correct_common_ocr_errors(text)
    text = fix_spacing_issues(text)

    return text
