# src/classifier.py
"""
Figure Classification using CLIP/ViT models.

🆕 BUGFIX: Added T4 GPU compatibility by detecting GPU type and using
float16 instead of bfloat16 for older GPU architectures.
"""

import torch
import cv2
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification


def _detect_gpu_dtype():
    """
    Detect the best dtype for the current GPU.
    
    T4, P100, K80, and other older GPUs don't support BFloat16 natively,
    which causes overflow errors. Use Float16 for these.
    
    Returns:
        torch.dtype: Best dtype for the current GPU
    """
    if not torch.cuda.is_available():
        return torch.float32
    
    device_name = torch.cuda.get_device_name(0).lower()
    
    # GPUs that don't support BFloat16 well
    older_gpus = ['t4', 'p100', 'k80', 'p4', 'p40', 'm60', 'v100']
    
    for gpu in older_gpus:
        if gpu in device_name:
            return torch.float16
    
    # A100, H100, and newer GPUs support BFloat16
    return torch.bfloat16


class FigureClassifier:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(FigureClassifier, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, 'model'):
            return

        if not torch.cuda.is_available():
            print("    - [WARNING] CUDA not available. Figure classification will be disabled.")
            self.processor = None
            self.model = None
            self.model_type = None
            self.dtype = torch.float32
            return

        # 🆕 BUGFIX: Detect GPU type and select appropriate dtype
        self.dtype = _detect_gpu_dtype()
        dtype_name = "float16" if self.dtype == torch.float16 else "bfloat16"
        print(f"    - Using {dtype_name} for GPU: {torch.cuda.get_device_name(0)}")

        # Try CLIP first (better for document figures)
        try:
            from transformers import CLIPProcessor, CLIPModel

            print("    - Initializing Figure Classifier (CLIP model for zero-shot)...")
            self.model_type = 'clip'
            model_id = "openai/clip-vit-base-patch32"
            self.processor = CLIPProcessor.from_pretrained(model_id)
            
            # 🆕 BUGFIX: Load model with GPU-appropriate dtype
            self.model = CLIPModel.from_pretrained(
                model_id,
                torch_dtype=self.dtype
            ).to("cuda")

            # Define document-specific labels for zero-shot classification
            self.candidate_labels = [
                "a bar chart with vertical or horizontal bars",
                "a line graph showing trends over time",
                "a pie chart showing percentages",
                "a scatter plot with data points",
                "a flowchart or process diagram",
                "a photograph or image",
                "a map or geographic visualization",
                "a table or data grid",
                "an organizational chart",
                "a technical diagram or schematic",
                "a logo or icon",
                "a signature or handwriting sample",
            ]

            self.label_to_kind = {
                "a bar chart with vertical or horizontal bars": "bar_chart",
                "a line graph showing trends over time": "line_chart",
                "a pie chart showing percentages": "pie_chart",
                "a scatter plot with data points": "scatter_plot",
                "a flowchart or process diagram": "flowchart",
                "a photograph or image": "photo",
                "a map or geographic visualization": "map",
                "a table or data grid": "table_image",
                "an organizational chart": "org_chart",
                "a technical diagram or schematic": "diagram",
                "a logo or icon": "logo",
                "a signature or handwriting sample": "signature",
            }

            print("    - ✅ Figure Classifier (CLIP) initialized successfully on GPU.")

        except Exception as e:
            print(f"    - Failed to load CLIP, falling back to ViT: {e}")
            self._init_vit_fallback()

    def _init_vit_fallback(self):
        """Fall back to ViT if CLIP fails."""
        try:
            print("    - Initializing Figure Classifier (ViT model)...")
            self.model_type = 'vit'
            model_id = "google/vit-base-patch16-224"
            self.processor = ViTImageProcessor.from_pretrained(model_id)
            
            # 🆕 BUGFIX: Use GPU-appropriate dtype
            self.model = ViTForImageClassification.from_pretrained(
                model_id,
                torch_dtype=self.dtype
            ).to("cuda")
            
            print("    - ✅ Figure Classifier (ViT) initialized successfully on GPU.")
        except Exception as e:
            print(f"    - [ERROR] Failed to initialize Figure Classifier: {e}")
            self.processor = None
            self.model = None
            self.model_type = None

    def _map_label_to_kind(self, label: str) -> str:
        """
        ENHANCED: Maps ImageNet labels to simple types needed by our project.
        Now includes many more keywords for better coverage.
        """
        label = label.lower()
        
        # Bar charts and column charts
        if any(keyword in label for keyword in [
            'bar chart', 'column chart', 'bar graph', 'histogram', 
            'vertical bar', 'horizontal bar', 'barplot'
        ]):
            return 'bar_chart'
        
        # Line graphs and line charts
        if any(keyword in label for keyword in [
            'line graph', 'line chart', 'time series', 'trend line',
            'line plot', 'curve', 'timeline'
        ]):
            return 'line_chart'
        
        # Pie charts
        if any(keyword in label for keyword in [
            'pie chart', 'pie graph', 'donut chart', 'circular chart',
            'percentage chart'
        ]):
            return 'pie_chart'
        
        # Scatter plots
        if any(keyword in label for keyword in [
            'scatter plot', 'scatter chart', 'scatterplot', 'point cloud',
            'dot plot', 'bubble chart'
        ]):
            return 'scatter_plot'
        
        # Maps and geographic visualizations
        if any(keyword in label for keyword in [
            'map', 'atlas', 'globe', 'cartography', 'geographic',
            'topographic', 'world map', 'street map', 'satellite'
        ]):
            return 'map'
        
        # Photos and photographs
        if any(keyword in label for keyword in [
            'photo', 'photograph', 'picture', 'image', 'snapshot',
            'digital camera', 'camera', 'portrait', 'landscape photo'
        ]):
            return 'photo'
        
        # Flowcharts and diagrams
        if any(keyword in label for keyword in [
            'flowchart', 'flow chart', 'workflow', 'process diagram',
            'decision tree', 'organizational chart', 'org chart',
            'hierarchy', 'sequence diagram', 'block diagram'
        ]):
            return 'flowchart'
        
        # Network diagrams
        if any(keyword in label for keyword in [
            'network', 'graph network', 'node', 'connection',
            'circuit', 'wiring', 'topology'
        ]):
            return 'network_diagram'
        
        # Architecture and blueprints
        if any(keyword in label for keyword in [
            'blueprint', 'architecture', 'floor plan', 'schematic',
            'technical drawing', 'engineering drawing', 'cad'
        ]):
            return 'blueprint'
        
        # Mathematical plots and functions
        if any(keyword in label for keyword in [
            'plot', 'graph', 'function', 'mathematical', 'equation',
            'coordinate', 'axis', 'cartesian'
        ]):
            return 'plot'
        
        # Tables (sometimes detected as images)
        if any(keyword in label for keyword in [
            'table', 'spreadsheet', 'grid', 'matrix', 'data table'
        ]):
            return 'table_image'
        
        # Diagrams (general)
        if any(keyword in label for keyword in [
            'diagram', 'illustration', 'schematic', 'chart',
            'infographic', 'visualization'
        ]):
            return 'diagram'
        
        # Scientific visualizations
        if any(keyword in label for keyword in [
            'microscope', 'telescope', 'x-ray', 'mri', 'scan',
            'spectrogram', 'oscilloscope', 'waveform', 'spectrum'
        ]):
            return 'scientific_viz'
        
        # Default fallback
        return 'figure'

    def classify(self, image_roi) -> str:
        """
        Classifies a given image ROI using CLIP (zero-shot) or ViT.
        
        🆕 BUGFIX: Added proper dtype handling for T4 GPU compatibility.
        """
        if self.model is None or self.processor is None:
            return "figure"

        try:
            # Validate input
            if image_roi is None or image_roi.size == 0:
                return "figure"
            
            # Convert BGR to RGB PIL Image
            if len(image_roi.shape) == 2:
                # Grayscale
                image_rgb = Image.fromarray(image_roi).convert('RGB')
            else:
                image_rgb = Image.fromarray(cv2.cvtColor(image_roi, cv2.COLOR_BGR2RGB))

            if self.model_type == 'clip':
                # Zero-shot classification with CLIP
                inputs = self.processor(
                    text=self.candidate_labels,
                    images=image_rgb,
                    return_tensors="pt",
                    padding=True
                )
                
                # 🆕 BUGFIX: Move inputs to GPU with correct dtype
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
                
                # Convert pixel values to the correct dtype
                if 'pixel_values' in inputs:
                    inputs['pixel_values'] = inputs['pixel_values'].to(self.dtype)

                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)

                best_idx = probs.argmax().item()
                best_label = self.candidate_labels[best_idx]

                return self.label_to_kind.get(best_label, "figure")

            else:  # ViT classification (fallback)
                inputs = self.processor(images=image_rgb, return_tensors="pt")
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
                
                # 🆕 BUGFIX: Convert to correct dtype
                if 'pixel_values' in inputs:
                    inputs['pixel_values'] = inputs['pixel_values'].to(self.dtype)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                logits = outputs.logits

                predicted_class_idx = logits.argmax(-1).item()
                predicted_label = self.model.config.id2label[predicted_class_idx]

                kind = self._map_label_to_kind(predicted_label)
                return kind

        except (ValueError, RuntimeError, OverflowError) as e:
            # 🆕 BUGFIX: Handle BFloat16/dtype errors gracefully
            error_str = str(e).lower()
            if 'bfloat16' in error_str or 'overflow' in error_str or 'dtype' in error_str:
                # Silently fall back to heuristic
                return "figure"
            print(f"    - Error during figure classification: {e}")
            return "figure"
        except Exception as e:
            print(f"    - Error during figure classification: {e}")
            return "figure"
