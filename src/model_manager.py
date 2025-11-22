# src/model_manager.py

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

try:
    from .handwriting_detector import HandwritingDetector
    HANDWRITING_DETECTOR_AVAILABLE = True
except ImportError:
    HANDWRITING_DETECTOR_AVAILABLE = False


class ModelManager:
    _instance = None
    _initialized = False
    
    # Model instances
    figure_classifier = None
    description_generator = None
    paddle_ocr = None
    trocr_processor = None
    trocr_model = None
    layout_model = None
    handwriting_detector = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance
    
    @classmethod
    def initialize(cls, load_all=True):
        if cls._initialized:
            print("ModelManager already initialized.")
            return
        
        print("\n" + "="*70)
        print("INITIALIZING MODEL MANAGER")
        print("Pre-loading all AI models to improve per-page processing speed...")
        print("="*70)

        instance = cls()

        # 1. Figure Classifier (ViT)
        print("\n[1/6] Loading Figure Classifier (ViT)...")
        try:
            instance.figure_classifier = FigureClassifier()
            print("      ✅ Figure Classifier loaded")
        except Exception as e:
            print(f"      ❌ Failed to load Figure Classifier: {e}")

        # 2. Description Generator (Phi-3)
        print("\n[2/6] Loading Description Generator (Phi-3)...")
        try:
            instance.description_generator = DescriptionGenerator()
            print("      ✅ Description Generator loaded")
        except Exception as e:
            print(f"      ❌ Failed to load Description Generator: {e}")

        # 3. PaddleOCR
        print("\n[3/6] Loading PaddleOCR...")
        if PADDLE_AVAILABLE and load_all:
            try:
                instance.paddle_ocr = PaddleOCR(use_angle_cls=True, lang='latin', show_log=False)
                print("      ✅ PaddleOCR loaded")
            except Exception as e:
                print(f"      ❌ Failed to load PaddleOCR: {e}")
        else:
            print("      ⊘ PaddleOCR not available or skipped")

        # 4. TrOCR
        print("\n[4/6] Loading TrOCR (Handwriting Expert)...")
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
        print("\n[5/6] Loading LayoutParser Model...")
        if LAYOUTPARSER_AVAILABLE and load_all:
            try:
                # Note: PaddleDetection models may have lower accuracy than Detectron2
                # Using PPYOLOv2 with adjusted threshold to reduce class collapse
                instance.layout_model = lp.models.PaddleDetectionLayoutModel(
                    config_path="lp://PubLayNet/ppyolov2_r50vd_dcn_365e/config",
                    label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
                    enforce_cpu=False,
                    extra_config={
                        'threshold': 0.3,  # Lower threshold for better recall across all classes
                    }
                )
                print("      ✅ LayoutParser Model loaded (threshold=0.3 for better class diversity)")
            except Exception as e:
                print(f"      ⊘ LayoutParser not available or skipped (Will fallback to Tesseract): {e}")
                instance.layout_model = None
        else:
            print("      ⊘ LayoutParser not available or skipped")

        # 6. Handwriting Detector
        print("\n[6/6] Loading Handwriting Detector...")
        if HANDWRITING_DETECTOR_AVAILABLE and load_all:
            try:
                instance.handwriting_detector = HandwritingDetector()
                print("      ✅ Handwriting Detector loaded")
            except Exception as e:
                print(f"      ❌ Failed to load Handwriting Detector: {e}")
        else:
            print("      ⊘ Handwriting Detector not available or skipped")
        
        cls._initialized = True
        print("\n" + "="*70)
        print("MODEL MANAGER INITIALIZATION COMPLETE")
        print("="*70 + "\n")
    
    @classmethod
    def get_figure_classifier(cls):
        return cls._instance.figure_classifier if cls._instance else None
    
    @classmethod
    def get_description_generator(cls):
        return cls._instance.description_generator if cls._instance else None
    
    @classmethod
    def get_paddle_ocr(cls):
        return cls._instance.paddle_ocr if cls._instance else None
    
    @classmethod
    def get_trocr(cls):
        if cls._instance:
            return cls._instance.trocr_processor, cls._instance.trocr_model
        return None, None
    
    @classmethod
    def get_layout_model(cls):
        return cls._instance.layout_model if cls._instance else None

    @classmethod
    def get_handwriting_detector(cls):
        return cls._instance.handwriting_detector if cls._instance else None