# src/dictionary_utils.py
"""
🆕 Dictionary-Based Fuzzy Matching for OCR Correction (Feature 4)

This module provides deterministic spelling correction for OCR output using
Levenshtein distance and dictionary lookups. Unlike LLM-based correction,
this approach is:
- Deterministic (same input = same output)
- Fast (no GPU required)
- Safe (won't introduce hallucinations)
- Configurable (adjustable thresholds and dictionaries)

Usage:
    from src.dictionary_utils import apply_fuzzy_corrections
    
    corrected_text = apply_fuzzy_corrections("COVIDVERTANCE -> INADVERTENCE")

Safety Features:
- Preserves alphanumeric codes (serial numbers, dates, etc.)
- Minimum word length requirement
- Strict similarity threshold
- Domain-specific dictionary support
"""

import re
import os
from typing import Optional, Set, List, Tuple, Dict
from functools import lru_cache

# Import config
try:
    from . import config
except ImportError:
    import config

# =============================================================================
#  DICTIONARY MANAGEMENT
# =============================================================================

# Global dictionary cache
_dictionary: Optional[Set[str]] = None
_symspell_instance = None
SYMSPELL_AVAILABLE = False

# Try to import SymSpell for faster fuzzy matching
try:
    from symspellpy import SymSpell, Verbosity
    SYMSPELL_AVAILABLE = True
except ImportError:
    SYMSPELL_AVAILABLE = False
    print("Warning: symspellpy not found. Using thefuzz for fuzzy matching (slower).")

# Try to import thefuzz as fallback
try:
    from thefuzz import fuzz
    THEFUZZ_AVAILABLE = True
except ImportError:
    THEFUZZ_AVAILABLE = False
    print("Warning: thefuzz not found. Fuzzy matching will be disabled.")


def _get_default_dictionary_path() -> str:
    """Get the path to the default English dictionary."""
    # Check for bundled dictionary
    module_dir = os.path.dirname(os.path.abspath(__file__))
    bundled_path = os.path.join(module_dir, 'data', 'english_dictionary.txt')
    
    if os.path.exists(bundled_path):
        return bundled_path
    
    # Check for system dictionary (Linux)
    system_paths = [
        '/usr/share/dict/words',
        '/usr/share/dict/american-english',
        '/usr/share/dict/british-english',
    ]
    
    for path in system_paths:
        if os.path.exists(path):
            return path
    
    return None


def load_dictionary(custom_path: str = None) -> Set[str]:
    """
    Load the dictionary from file.
    
    Args:
        custom_path: Path to custom dictionary file (one word per line)
    
    Returns:
        Set of valid words (lowercase)
    """
    global _dictionary
    
    if _dictionary is not None and custom_path is None:
        return _dictionary
    
    words = set()
    
    # Load default dictionary
    default_path = _get_default_dictionary_path()
    if default_path and os.path.exists(default_path):
        try:
            with open(default_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    word = line.strip().lower()
                    if word and len(word) >= 2:
                        words.add(word)
            print(f"    - Loaded {len(words)} words from default dictionary.")
        except Exception as e:
            print(f"    - Failed to load default dictionary: {e}")
    
    # Load custom dictionary
    custom_config_path = custom_path or getattr(config, 'FUZZY_CUSTOM_DICTIONARY', None)
    if custom_config_path and os.path.exists(custom_config_path):
        try:
            with open(custom_config_path, 'r', encoding='utf-8', errors='ignore') as f:
                custom_count = 0
                for line in f:
                    word = line.strip().lower()
                    if word:
                        words.add(word)
                        custom_count += 1
            print(f"    - Added {custom_count} words from custom dictionary.")
        except Exception as e:
            print(f"    - Failed to load custom dictionary: {e}")
    
    # Add common OCR-specific words and abbreviations
    ocr_specific = {
        'fax', 'tel', 'phone', 'email', 'attn', 'ref', 'cc', 'bcc',
        're', 'fw', 'fwd', 'enc', 'encl', 'pg', 'pgs', 'yr', 'yrs',
        'inc', 'corp', 'llc', 'ltd', 'plc', 'dept', 'div', 'ext',
        'mr', 'mrs', 'ms', 'dr', 'jr', 'sr', 'phd', 'md', 'esq',
        'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug',
        'sep', 'oct', 'nov', 'dec', 'mon', 'tue', 'wed', 'thu',
        'fri', 'sat', 'sun', 'am', 'pm', 'est', 'pst', 'cst', 'mst'
    }
    words.update(ocr_specific)
    
    # If no dictionary found, create a minimal fallback
    if len(words) < 1000:
        print("    - ⚠️ Dictionary is small. Creating minimal fallback...")
        words.update(_get_minimal_dictionary())
    
    _dictionary = words
    return words


def _get_minimal_dictionary() -> Set[str]:
    """Return a minimal dictionary of common English words."""
    # Top 1000 most common English words
    common_words = {
        'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
        'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
        'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
        'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what',
        'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me',
        'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know', 'take',
        'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see', 'other',
        'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also',
        'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way',
        'even', 'new', 'want', 'because', 'any', 'these', 'give', 'day', 'most', 'us',
        'please', 'thank', 'thanks', 'dear', 'sincerely', 'regards', 'attention',
        'subject', 'date', 'from', 'number', 'page', 'pages', 'copy', 'copies',
        'attached', 'enclosed', 'following', 'regarding', 'reference', 'request',
        'response', 'information', 'document', 'documents', 'office', 'company',
        'address', 'telephone', 'message', 'urgent', 'confidential', 'private',
        'sender', 'recipient', 'signature', 'signed', 'approved', 'reviewed',
        'meeting', 'schedule', 'report', 'invoice', 'payment', 'total', 'amount',
        'price', 'cost', 'fee', 'tax', 'shipping', 'order', 'customer', 'client',
        'account', 'balance', 'due', 'paid', 'received', 'sent', 'delivered',
        'inadvertence', 'accordance', 'compliance', 'pursuant', 'regarding',
        'whereas', 'therefore', 'hereby', 'hereafter', 'herein', 'thereof'
    }
    return common_words


def _initialize_symspell():
    """Initialize SymSpell for fast fuzzy matching."""
    global _symspell_instance
    
    if not SYMSPELL_AVAILABLE or not getattr(config, 'USE_SYMSPELL', True):
        return None
    
    if _symspell_instance is not None:
        return _symspell_instance
    
    try:
        max_edit_distance = getattr(config, 'FUZZY_MAX_EDIT_DISTANCE', 2)
        sym_spell = SymSpell(max_dictionary_edit_distance=max_edit_distance)
        
        # Load dictionary
        dictionary = load_dictionary()
        
        # Add words to SymSpell
        for word in dictionary:
            sym_spell.create_dictionary_entry(word, 1)
        
        _symspell_instance = sym_spell
        print(f"    - ✅ SymSpell initialized with {len(dictionary)} words.")
        return sym_spell
        
    except Exception as e:
        print(f"    - ⚠️ Failed to initialize SymSpell: {e}")
        return None


# =============================================================================
#  WORD CLASSIFICATION (Safety Checks)
# =============================================================================

def is_alphanumeric_code(word: str) -> bool:
    """
    Check if a word appears to be an alphanumeric code that should NOT be corrected.
    
    This protects:
    - Serial numbers (e.g., "ABC12345")
    - Reference codes (e.g., "REF-2024-001")
    - Dates (e.g., "01/15/2024", "2024-01-15")
    - Phone numbers (e.g., "(555) 123-4567")
    - ZIP codes (e.g., "12345-6789")
    - Email addresses
    - URLs
    """
    if not word:
        return False
    
    # Check for mixed letters and digits
    has_letters = any(c.isalpha() for c in word)
    has_digits = any(c.isdigit() for c in word)
    
    if has_letters and has_digits:
        return True
    
    # Check for patterns that suggest codes
    code_patterns = [
        r'^\d{2,}$',                    # Pure numbers (2+ digits)
        r'^[A-Z]{2,}\d+$',              # Letters followed by numbers
        r'^\d+[A-Z]{2,}$',              # Numbers followed by letters
        r'^[A-Z0-9]{2,}-[A-Z0-9]{2,}$', # Hyphenated codes
        r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$',  # Dates
        r'^\(\d{3}\)\s?\d{3}-\d{4}$',   # Phone numbers
        r'^\d{3}-\d{2}-\d{4}$',         # SSN-like
        r'^\d{5}(-\d{4})?$',            # ZIP codes
        r'^[A-Z]{1,2}\d{1,2}[A-Z]?\s?\d[A-Z]{2}$',  # UK postcodes
        r'@',                            # Email addresses
        r'^https?://',                   # URLs
        r'^\$[\d,]+\.?\d*$',            # Money amounts
        r'^#\d+$',                       # Reference numbers
    ]
    
    for pattern in code_patterns:
        if re.match(pattern, word, re.IGNORECASE):
            return True
    
    return False


def should_skip_word(word: str) -> bool:
    """
    Determine if a word should be skipped from fuzzy correction.
    
    Skip if:
    - Too short
    - Is an alphanumeric code
    - Is all uppercase (likely an acronym)
    - Contains special characters
    - Is already in dictionary
    """
    min_length = getattr(config, 'FUZZY_MIN_WORD_LENGTH', 4)
    skip_alphanumeric = getattr(config, 'FUZZY_SKIP_ALPHANUMERIC', True)
    
    # Length check
    if len(word) < min_length:
        return True
    
    # Alphanumeric code check
    if skip_alphanumeric and is_alphanumeric_code(word):
        return True
    
    # All uppercase (likely acronym)
    if word.isupper() and len(word) <= 5:
        return True
    
    # Contains non-letter characters (except apostrophe and hyphen)
    cleaned = re.sub(r"['-]", '', word)
    if not cleaned.isalpha():
        return True
    
    # Already in dictionary
    dictionary = load_dictionary()
    if word.lower() in dictionary:
        return True
    
    return False


# =============================================================================
#  FUZZY MATCHING CORE
# =============================================================================

@lru_cache(maxsize=10000)
def find_best_match_symspell(word: str) -> Tuple[Optional[str], float]:
    """
    Find the best dictionary match using SymSpell.
    
    Args:
        word: Word to find match for
    
    Returns:
        Tuple of (best_match, similarity_score) or (None, 0.0)
    """
    sym_spell = _initialize_symspell()
    if sym_spell is None:
        return None, 0.0
    
    max_edit_distance = getattr(config, 'FUZZY_MAX_EDIT_DISTANCE', 2)
    
    try:
        suggestions = sym_spell.lookup(
            word.lower(),
            Verbosity.CLOSEST,
            max_edit_distance=max_edit_distance
        )
        
        if suggestions:
            best = suggestions[0]
            # Calculate similarity score
            max_len = max(len(word), len(best.term))
            similarity = 1.0 - (best.distance / max_len)
            return best.term, similarity
        
        return None, 0.0
        
    except Exception:
        return None, 0.0


@lru_cache(maxsize=10000)
def find_best_match_thefuzz(word: str) -> Tuple[Optional[str], float]:
    """
    Find the best dictionary match using thefuzz (Levenshtein distance).
    
    This is slower than SymSpell but works as a fallback.
    
    Args:
        word: Word to find match for
    
    Returns:
        Tuple of (best_match, similarity_score) or (None, 0.0)
    """
    if not THEFUZZ_AVAILABLE:
        return None, 0.0
    
    dictionary = load_dictionary()
    threshold = getattr(config, 'FUZZY_SIMILARITY_THRESHOLD', 0.80) * 100
    
    word_lower = word.lower()
    best_match = None
    best_score = 0
    
    # For efficiency, only check words of similar length
    word_len = len(word)
    candidates = [w for w in dictionary if abs(len(w) - word_len) <= 2]
    
    for dict_word in candidates:
        score = fuzz.ratio(word_lower, dict_word)
        if score > best_score and score >= threshold:
            best_score = score
            best_match = dict_word
    
    if best_match:
        return best_match, best_score / 100.0
    
    return None, 0.0


def find_best_match(word: str) -> Tuple[Optional[str], float]:
    """
    Find the best dictionary match for a word.
    
    Uses SymSpell if available, falls back to thefuzz.
    
    Args:
        word: Word to find match for
    
    Returns:
        Tuple of (best_match, similarity_score) or (None, 0.0)
    """
    # Try SymSpell first (faster)
    if SYMSPELL_AVAILABLE and getattr(config, 'USE_SYMSPELL', True):
        match, score = find_best_match_symspell(word)
        if match:
            return match, score
    
    # Fall back to thefuzz
    if THEFUZZ_AVAILABLE:
        return find_best_match_thefuzz(word)
    
    return None, 0.0


# =============================================================================
#  MAIN CORRECTION FUNCTIONS
# =============================================================================

def correct_word(word: str) -> str:
    """
    Attempt to correct a single word using fuzzy dictionary matching.
    
    Args:
        word: Word to potentially correct
    
    Returns:
        Corrected word (preserving original case) or original if no correction found
    """
    # Skip words that shouldn't be corrected
    if should_skip_word(word):
        return word
    
    threshold = getattr(config, 'FUZZY_SIMILARITY_THRESHOLD', 0.80)
    
    # Find best match
    match, score = find_best_match(word)
    
    if match and score >= threshold:
        # Preserve original case pattern
        return _apply_case_pattern(word, match)
    
    return word


def _apply_case_pattern(original: str, replacement: str) -> str:
    """
    Apply the case pattern from the original word to the replacement.
    
    Examples:
        - "COVIDVERTANCE" -> "INADVERTENCE"
        - "Covidvertance" -> "Inadvertence"
        - "covidvertance" -> "inadvertence"
    """
    if original.isupper():
        return replacement.upper()
    elif original.istitle():
        return replacement.title()
    elif original.islower():
        return replacement.lower()
    else:
        # Mixed case - return lowercase match
        return replacement.lower()


def apply_fuzzy_corrections(text: str) -> str:
    """
    Apply fuzzy dictionary corrections to OCR text.
    
    This is the main entry point for OCR text correction.
    
    Args:
        text: OCR output text
    
    Returns:
        Text with typos corrected based on dictionary matching
    """
    if not getattr(config, 'USE_FUZZY_MATCHING', True):
        return text
    
    if not text:
        return text
    
    # Ensure dictionary is loaded
    load_dictionary()
    
    # Split into words while preserving delimiters
    # This regex captures words and the spaces/punctuation between them
    tokens = re.split(r'(\s+|[^\w]+)', text)
    
    corrected_tokens = []
    corrections_made = 0
    
    for token in tokens:
        # Only process word-like tokens
        if token and re.match(r'^\w+$', token):
            corrected = correct_word(token)
            if corrected != token:
                corrections_made += 1
            corrected_tokens.append(corrected)
        else:
            corrected_tokens.append(token)
    
    result = ''.join(corrected_tokens)
    
    if corrections_made > 0:
        print(f"    - Fuzzy matching: {corrections_made} correction(s) applied.")
    
    return result


def get_correction_suggestions(word: str, max_suggestions: int = 5) -> List[Tuple[str, float]]:
    """
    Get a list of possible corrections for a word with their scores.
    
    Useful for debugging or showing alternatives to users.
    
    Args:
        word: Word to get suggestions for
        max_suggestions: Maximum number of suggestions to return
    
    Returns:
        List of (suggestion, score) tuples, sorted by score descending
    """
    if should_skip_word(word):
        return [(word, 1.0)]
    
    suggestions = []
    
    # Try SymSpell
    if SYMSPELL_AVAILABLE:
        sym_spell = _initialize_symspell()
        if sym_spell:
            max_edit = getattr(config, 'FUZZY_MAX_EDIT_DISTANCE', 2)
            results = sym_spell.lookup(word.lower(), Verbosity.ALL, max_edit_distance=max_edit)
            for r in results[:max_suggestions]:
                max_len = max(len(word), len(r.term))
                score = 1.0 - (r.distance / max_len)
                suggestions.append((_apply_case_pattern(word, r.term), score))
    
    # Fall back to thefuzz if needed
    if not suggestions and THEFUZZ_AVAILABLE:
        dictionary = load_dictionary()
        word_lower = word.lower()
        word_len = len(word)
        candidates = [w for w in dictionary if abs(len(w) - word_len) <= 2]
        
        scored = []
        for dict_word in candidates:
            score = fuzz.ratio(word_lower, dict_word) / 100.0
            if score >= 0.5:  # Lower threshold for suggestions
                scored.append((dict_word, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        suggestions = [(_apply_case_pattern(word, w), s) for w, s in scored[:max_suggestions]]
    
    return suggestions if suggestions else [(word, 1.0)]


# =============================================================================
#  INITIALIZATION
# =============================================================================

def initialize():
    """
    Pre-initialize the dictionary and fuzzy matching engine.
    
    Call this during application startup to avoid delay on first correction.
    """
    print("  - Initializing fuzzy matching dictionary...")
    load_dictionary()
    
    if SYMSPELL_AVAILABLE and getattr(config, 'USE_SYMSPELL', True):
        _initialize_symspell()
    
    print("  - ✅ Fuzzy matching ready.")


# Auto-initialize when module is imported (if enabled)
if getattr(config, 'USE_FUZZY_MATCHING', True):
    # Lazy initialization - dictionary loaded on first use
    pass
