# src/model_manager.py

"""
Central Model Manager
Pre-loads all AI models at startup to avoid per-page initialization delays.
"""

from .classifier import FigureClassifier
from .description_generator import DescriptionGenerator
import torch

# Try to import optional dependencies
try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False

try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    TROCR_AVAILABLE = True
except ImportError:
    TROCR_AVAILABLE = False

try:
    import layoutparser as lp
    LAYOUTPARSER_AVAILABLE = True
except ImportError:
    LAYOUTPARSER_AVAILABLE = False


class ModelManager:
    """
    Singleton class that pre-loads and manages all AI models.
    Call ModelManager.initialize() once at startup to load all models.
    """
    _instance = None
    _initialized = False
    
    # Model instances
    figure_classifier = None
    description_generator = None
    paddle_ocr = None
    trocr_processor = None
    trocr_model = None
    layout_model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance
    
    @classmethod
    def initialize(cls, load_all=True):
        """
        Pre-loads all available AI models.
        
        Args:
            load_all: If True, loads all models. If False, loads only essential ones.
        """
        if cls._initialized:
            print("ModelManager already initialized.")
            return
        
        print("\n" + "="*70)
        print("INITIALIZING MODEL MANAGER")
        print("Pre-loading all AI models to improve per-page processing speed...")
        print("="*70)
        
        instance = cls()
        
        # 1. Figure Classifier (ViT)
        print("\n[1/5] Loading Figure Classifier (ViT)...")
        try:
            instance.figure_classifier = FigureClassifier()
            print("      ✅ Figure Classifier loaded")
        except Exception as e:
            print(f"      ❌ Failed to load Figure Classifier: {e}")
        
        # 2. Description Generator (Phi-3)
        print("\n[2/5] Loading Description Generator (Phi-3)...")
        try:
            instance.description_generator = DescriptionGenerator()
            print("      ✅ Description Generator loaded")
        except Exception as e:
            print(f"      ❌ Failed to load Description Generator: {e}")
        
        # 3. PaddleOCR
        print("\n[3/5] Loading PaddleOCR...")
        if PADDLE_AVAILABLE and load_all:
            try:
                instance.paddle_ocr = PaddleOCR(use_angle_cls=True, lang='latin', show_log=False)
                print("      ✅ PaddleOCR loaded")
            except Exception as e:
                print(f"      ❌ Failed to load PaddleOCR: {e}")
        else:
            print("      ⊘ PaddleOCR not available or skipped")
        
        # 4. TrOCR
        print("\n[4/5] Loading TrOCR (Handwriting Expert)...")
        if TROCR_AVAILABLE and torch.cuda.is_available() and load_all:
            try:
                model_id = "microsoft/trocr-base-handwritten"
                instance.trocr_processor = TrOCRProcessor.from_pretrained(model_id)
                instance.trocr_model = VisionEncoderDecoderModel.from_pretrained(model_id).to("cuda")
                print("      ✅ TrOCR loaded")
            except Exception as e:
                print(f"      ❌ Failed to load TrOCR: {e}")
        else:
            print("      ⊘ TrOCR not available, CUDA unavailable, or skipped")
        
        # 5. LayoutParser
        print("\n[5/5] Loading LayoutParser Model...")
        if LAYOUTPARSER_AVAILABLE and load_all:
            try:
                instance.layout_model = lp.models.PaddleDetectionLayoutModel(
                    config_path="lp://PubLayNet/ppyolov2_r50vd_dcn_365e_publaynet/config",
                    threshold=0.5,
                    label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
                    enforce_cpu=False,
                    enable_mkldnn=True
                )
                print("      ✅ LayoutParser Model loaded")
            except Exception as e:
                print(f"      ❌ Failed to load LayoutParser: {e}")
        else:
            print("      ⊘ LayoutParser not available or skipped")
        
        cls._initialized = True
        print("\n" + "="*70)
        print("MODEL MANAGER INITIALIZATION COMPLETE")
        print("="*70 + "\n")
    
    @classmethod
    def get_figure_classifier(cls):
        """Returns the pre-loaded figure classifier."""
        return cls._instance.figure_classifier if cls._instance else None
    
    @classmethod
    def get_description_generator(cls):
        """Returns the pre-loaded description generator."""
        return cls._instance.description_generator if cls._instance else None
    
    @classmethod
    def get_paddle_ocr(cls):
        """Returns the pre-loaded PaddleOCR instance."""
        return cls._instance.paddle_ocr if cls._instance else None
    
    @classmethod
    def get_trocr(cls):
        """Returns the pre-loaded TrOCR processor and model."""
        if cls._instance:
            return cls._instance.trocr_processor, cls._instance.trocr_model
        return None, None
    
    @classmethod
    def get_layout_model(cls):
        """Returns the pre-loaded LayoutParser model."""
        return cls._instance.layout_model if cls._instance else None