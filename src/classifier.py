# src/classifier.py

import torch
import cv2
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