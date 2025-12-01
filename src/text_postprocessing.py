# src/text_postprocessing.py
"""
Text Post-Processing Module for OCR Output

This module provides text cleaning and correction functions for OCR output,
including common OCR error fixes and spacing normalization.
"""

import re
from typing import Dict, List, Tuple

# =============================================================================
#  COMMON OCR ERROR MAPPINGS
# =============================================================================

# Common character substitution errors in OCR
CHAR_SUBSTITUTIONS: Dict[str, str] = {
    # l/1/I confusion
    'l1': 'li',
    '1l': 'il',
    'I1': 'Il',
    '1I': 'II',
    
    # 0/O confusion
    '0O': 'OO',
    'O0': 'O0',
    
    # rn/m confusion
    'rn': 'm',  # Only in specific contexts
    
    # Common OCR artifacts
    '|': 'l',
    '¦': 'l',
    '!': 'l',  # Only when clearly misread
    
    # Punctuation fixes
    ',,': ',',
    '..': '.',
    '  ': ' ',
}

# Context-aware substitutions (only apply when surrounded by letters)
CONTEXT_SUBSTITUTIONS: List[Tuple[str, str, str]] = [
    # (pattern, replacement, context_type)
    (r'(\w)1(\w)', r'\1l\2', 'word'),  # 1 -> l in words
    (r'(\w)0(\w)', r'\1o\2', 'word'),  # 0 -> o in words (lowercase context)
    (r'(\w)\|(\w)', r'\1l\2', 'word'),  # | -> l in words
]

# Words that are frequently misrecognized
COMMON_CORRECTIONS: Dict[str, str] = {
    'teh': 'the',
    'adn': 'and',
    'hte': 'the',
    'fo': 'of',
    'ot': 'to',
    'nad': 'and',
    'tme': 'the',
    'rnessage': 'message',
    'tirne': 'time',
    'nurnber': 'number',
    'cornpany': 'company',
    'frorn': 'from',
    'narne': 'name',
    'sarne': 'same',
    'corne': 'come',
    'horne': 'home',
    'sorne': 'some',
}


# =============================================================================
#  TEXT CLEANING FUNCTIONS
# =============================================================================

def fix_spacing_issues(text: str) -> str:
    """
    Fix common spacing issues in OCR output.
    
    Handles:
    - Multiple spaces
    - Spaces before punctuation
    - Missing spaces after punctuation
    - Line break artifacts
    """
    if not text:
        return text
    
    # Remove multiple spaces
    text = re.sub(r' {2,}', ' ', text)
    
    # Remove spaces before punctuation
    text = re.sub(r'\s+([.,;:!?)])', r'\1', text)
    
    # Add space after punctuation if missing (except for decimals and abbreviations)
    text = re.sub(r'([.,;:!?])([A-Za-z])', r'\1 \2', text)
    
    # Fix common line break artifacts
    text = re.sub(r'-\s*\n\s*', '', text)  # Hyphenated line breaks
    text = re.sub(r'\n+', ' ', text)  # Multiple newlines to space
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text


def apply_char_substitutions(text: str) -> str:
    """
    Apply character-level substitutions for common OCR errors.
    
    Note: This is conservative and only applies safe substitutions.
    """
    if not text:
        return text
    
    # Apply context-aware substitutions
    for pattern, replacement, context_type in CONTEXT_SUBSTITUTIONS:
        if context_type == 'word':
            text = re.sub(pattern, replacement, text)
    
    return text


def apply_word_corrections(text: str) -> str:
    """
    Apply word-level corrections for known OCR errors.
    
    Only corrects words that are very commonly misrecognized.
    """
    if not text:
        return text
    
    words = text.split()
    corrected_words = []
    
    for word in words:
        # Preserve punctuation attached to word
        prefix = ''
        suffix = ''
        
        # Extract leading punctuation
        match = re.match(r'^([^\w]*)(.*?)([^\w]*)$', word)
        if match:
            prefix, core, suffix = match.groups()
        else:
            core = word
        
        # Check if core word needs correction
        core_lower = core.lower()
        if core_lower in COMMON_CORRECTIONS:
            # Preserve case
            if core.isupper():
                core = COMMON_CORRECTIONS[core_lower].upper()
            elif core.istitle():
                core = COMMON_CORRECTIONS[core_lower].title()
            else:
                core = COMMON_CORRECTIONS[core_lower]
        
        corrected_words.append(prefix + core + suffix)
    
    return ' '.join(corrected_words)


def remove_ocr_artifacts(text: str) -> str:
    """
    Remove common OCR artifacts and noise.
    
    Handles:
    - Random single characters
    - Repeated punctuation
    - Box drawing characters
    - Control characters
    """
    if not text:
        return text
    
    # Remove box drawing and special characters
    text = re.sub(r'[─│┌┐└┘├┤┬┴┼═║╒╓╔╕╖╗╘╙╚╛╜╝╞╟╠╡╢╣╤╥╦╧╨╩╪╫╬]', '', text)
    
    # Remove control characters (except newlines and tabs)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    
    # Remove isolated single characters (noise)
    text = re.sub(r'\s+[^\w\s]\s+', ' ', text)
    
    # Remove repeated punctuation
    text = re.sub(r'([.,;:!?])\1+', r'\1', text)
    
    return text


def normalize_quotes(text: str) -> str:
    """
    Normalize various quote characters to standard ASCII quotes.
    """
    if not text:
        return text
    
    # Smart quotes to straight quotes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    text = text.replace('«', '"').replace('»', '"')
    text = text.replace('‹', "'").replace('›', "'")
    
    return text


def fix_number_formatting(text: str) -> str:
    """
    Fix common number formatting issues from OCR.
    
    Handles:
    - Spaces in numbers
    - O/0 confusion in numeric contexts
    """
    if not text:
        return text
    
    # Fix spaces within numbers (e.g., "1 234" -> "1234")
    text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)
    
    # Fix O in clearly numeric sequences (e.g., "2O24" -> "2024")
    text = re.sub(r'(\d)O(\d)', r'\g<1>0\2', text)
    text = re.sub(r'(\d)O([,.\s]|$)', r'\g<1>0\2', text)
    
    return text


# =============================================================================
#  MAIN CORRECTION PIPELINE
# =============================================================================

def apply_text_corrections(text: str, aggressive: bool = False) -> str:
    """
    Apply all text corrections to OCR output.
    
    This is the main entry point for text post-processing.
    
    Args:
        text: Raw OCR output text
        aggressive: If True, apply more aggressive corrections (may over-correct)
    
    Returns:
        Cleaned and corrected text
    """
    if not text:
        return text
    
    # Step 1: Remove obvious artifacts
    text = remove_ocr_artifacts(text)
    
    # Step 2: Normalize quotes
    text = normalize_quotes(text)
    
    # Step 3: Fix spacing
    text = fix_spacing_issues(text)
    
    # Step 4: Fix number formatting
    text = fix_number_formatting(text)
    
    # Step 5: Apply character substitutions
    if aggressive:
        text = apply_char_substitutions(text)
    
    # Step 6: Apply word corrections
    text = apply_word_corrections(text)
    
    # Final cleanup
    text = fix_spacing_issues(text)
    
    return text


def get_correction_stats(original: str, corrected: str) -> Dict:
    """
    Get statistics about corrections made.
    
    Args:
        original: Original text
        corrected: Corrected text
    
    Returns:
        Dictionary with correction statistics
    """
    if not original or not corrected:
        return {'changes': 0, 'original_length': len(original or ''), 'corrected_length': len(corrected or '')}
    
    original_words = set(original.lower().split())
    corrected_words = set(corrected.lower().split())
    
    added = corrected_words - original_words
    removed = original_words - corrected_words
    
    return {
        'original_length': len(original),
        'corrected_length': len(corrected),
        'words_changed': len(added) + len(removed),
        'words_added': list(added)[:10],  # Limit for readability
        'words_removed': list(removed)[:10],
    }
