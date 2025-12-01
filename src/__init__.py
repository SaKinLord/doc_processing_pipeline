# src/__init__.py
"""
Document Processing Pipeline - Advanced OCR and Document Analysis

This package provides a comprehensive document processing pipeline with:
- Multi-engine OCR (Tesseract, PaddleOCR, TrOCR, Surya)
- AI-powered image super-resolution
- Vision-Language Model integration
- Deterministic fuzzy matching for typo correction
- Layout analysis and table extraction
- Handwriting detection and recognition

New Features (v2.0):
- 🆕 AI Super-Resolution for low-quality documents (EDSR, ESPCN, FSRCNN, LapSRN)
- 🆕 Surya OCR for superior text line detection
- 🆕 VLM (Qwen-VL-Chat, LLaVA) for complex form understanding
- 🆕 Deterministic fuzzy matching with SymSpell/thefuzz

Usage:
    from src import config
    from src.pipeline import DocumentProcessor
    from src.ocr import ocr_smart, ocr_routed
    from src.vlm_manager import VLMManager
    from src.dictionary_utils import apply_fuzzy_corrections
"""

__version__ = "2.0.0"
__author__ = "Document Processing Team"

# Feature flags summary
from . import config

def print_feature_status():
    """Print the status of all advanced features."""
    config.print_feature_status()
