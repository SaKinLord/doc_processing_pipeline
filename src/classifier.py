# src/classifier.py

import torch
import cv2  # ADDED: Import OpenCV library
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification

class FigureClassifier:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(FigureClassifier, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, 'model'):
            return

        print("    - Initializing Figure Classifier (ViT model)...")
        if not torch.cuda.is_available():
            print("    - [WARNING] CUDA not available. Figure classification will be disabled.")
            self.processor = None
            self.model = None
            return

        try:
            model_id = "google/vit-base-patch16-224"
            self.processor = ViTImageProcessor.from_pretrained(model_id)
            self.model = ViTForImageClassification.from_pretrained(model_id).to("cuda")
            print("    - ✅ Figure Classifier initialized successfully on GPU.")
        except Exception as e:
            print(f"    - [ERROR] Failed to initialize Figure Classifier: {e}")
            self.processor = None
            self.model = None

    def _map_label_to_kind(self, label: str) -> str:
        """Maps ImageNet labels to simple types needed by our project."""
        label = label.lower()
        if 'bar chart' in label or 'column chart' in label:
            return 'bar_chart'
        if 'line graph' in label or 'line chart' in label:
            return 'line_chart'
        if 'pie chart' in label:
            return 'pie_chart'
        if 'map' in label:
            return 'map'
        if 'photo' in label or 'photograph' in label or 'digital camera' in label:
            return 'photo'
        if 'plot' in label or 'diagram' in label or 'graph' in label:
            return 'plot'
        return 'figure'

    def classify(self, image_roi) -> str:
        """Classifies a given image ROI and returns its type."""
        if self.model is None or self.processor is None:
            return "figure"

        try:
            image_rgb = Image.fromarray(cv2.cvtColor(image_roi, cv2.COLOR_BGR2RGB))
            
            inputs = self.processor(images=image_rgb, return_tensors="pt").to("cuda")
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            predicted_class_idx = logits.argmax(-1).item()
            predicted_label = self.model.config.id2label[predicted_class_idx]
            
            kind = self._map_label_to_kind(predicted_label)
            return kind
        except Exception as e:
            print(f"    - Error during figure classification: {e}")
            return "figure"