# Document AI Pipeline (Local & Self-Contained)

This project is a locally runnable, self-contained Document AI pipeline that processes multi-format documents (PDF, scanned images, photos, handwriting, etc.) and converts them into a structured JSON format. The system allows you to securely process your sensitive data on your own infrastructure without the need for external cloud APIs.

## ✨ Key Features

### Core Document Processing
*   **Multi-Format Support:** Processes common document and image formats like PDF, PNG, JPG, JPEG, BMP, and TIFF.
*   **Intelligent Layout Analysis:**
    *   Uses LayoutParser with PaddleDetection (PicoDet model) for advanced layout detection (lightweight & fast)
    *   Automatically falls back to Tesseract-based layout analysis if LayoutParser is unavailable
    *   Detects text blocks, titles, lists, tables, and figures with confidence scores
    *   Automatically detects columns and determines correct reading order

### Advanced OCR Capabilities
*   **🧠 Smart OCR Router with Handwriting Detection:**
    *   **Pre-Classification:** Automatically detects if text is printed or handwritten before OCR
    *   **Direct Routing:** Routes to optimal OCR engine (Tesseract for printed, TrOCR for handwritten)
    *   **Quality Scoring:** Multi-factor text quality assessment (fragmentation, character composition, noise patterns)
    *   **3-Stage Fallback Chain:** Tesseract → PaddleOCR → TrOCR with intelligent triggering based on confidence scores (<60%) and fragmentation analysis.
    *   **Noise Filtering:** Automatically discards OCR results with extremely low quality scores to prevent garbage data output.
    *   **~60% faster processing** by skipping unnecessary OCR stages
*   **Automatic Language Detection:**
    *   Uses langdetect library with confidence thresholding (98%)
    *   Supports multiple languages: English, Turkish, German, French, Spanish, Indonesian
    *   Language override option for multi-page documents

### Table & Figure Processing
*   **Advanced Table Structure Recognition:**
    *   **Merged Cell Detection:** Automatically detects rowspan/colspan by analyzing line continuity
    *   **Multiple Export Formats:** CSV, HTML (with proper span attributes), Markdown
    *   **Bordered Tables:** Detects using Hough Transform with morphological enhancement
    *   **Borderless Tables:** Uses alignment-based detection from text block positioning
    *   Validates table structure to distinguish between data tables and forms
*   **AI-Powered Figure Understanding:**
    *   **ViT (Vision Transformer):** Classifies figures into 14+ types (bar_chart, line_chart, pie_chart, photo, map, flowchart, etc.)
    *   **Chart Data Extraction:** Extracts numerical data from bar charts, line charts, pie charts, and scatter plots
    *   Intelligent caption detection (searches below and above figures)
    *   Filters out false positives using content density analysis
    *   **Microsoft Phi-3:** Generates natural language descriptions using 4-bit quantization

### Form & Field Extraction
*   **Hybrid Rule-Based + LLM Field Extraction:**
    *   **Rule-Based:** Fast regex matching with multi-directional search (right, below, diagonal)
    *   **LLM-Enhanced:** Uses Phi-3 to extract fields that patterns miss
    *   **🛡️ Hallucination Prevention System:**
        *   **Strict Verification:** Every value extracted by LLM is cross-verified against the source text using fuzzy substring matching.
        *   **Blacklist Filtering:** Automatically blocks generic placeholders (e.g., "John Smith", "ACME Corp") and common LLM hallucinations.
        *   **Zero-Hallucination Guarantee:** If a field is not explicitly present in the document, it returns null instead of inventing data.
    *   Supports common form fields: fax, phone, date, to, from, subject, pages
    *   Automatic data normalization (dates to YYYY-MM-DD, phone numbers to international format)
    *   Tracks extraction method (rule_based or llm)

### Image Preprocessing
*   **Adaptive Quality-Based Preprocessing:**
    *   **Automatic Quality Analysis:** Analyzes Laplacian variance, brightness, contrast
    *   **Dynamic Parameter Selection:** Adjusts denoising and enhancement based on image quality
    *   **Quality Presets:** clean, normal, noisy, blurry with optimized parameters
    *   Automatic deskewing using Hough Transform
    *   CLAHE (Contrast Limited Adaptive Histogram Equalization)
*   **Line Detection & Removal:** Detects and removes table lines for cleaner text extraction

### Performance & Optimization
*   **Model Manager:** Pre-loads all 6 AI models at startup (ViT, Phi-3, PaddleOCR, TrOCR, LayoutParser, Handwriting Detector)
*   **Parallel Processing:**
    *   **File-Level:** Process multiple files concurrently with ProcessPoolExecutor
    *   **Page-Level:** Process PDF pages in parallel with ThreadPoolExecutor (NEW!)
*   **GPU Acceleration:** Full CUDA support for TrOCR, ViT, Phi-3, and LayoutParser models
*   **Graph-Based Reading Order:** Topological sorting for correct text flow in complex layouts
*   **Smart Model Loading:** Optional `--no-preload` flag for faster startup with on-demand model loading

### Security & Privacy
*   **🚀 100% Local and Secure:** All models (TrOCR, ViT, Phi-3, PaddleOCR, LayoutParser) run on the local machine. Your data never leaves your premises.
*   **No External API Calls:** Complete offline operation after initial model downloads
*   **Packaged with Docker:** Delivered as a single, ready-to-run Docker image with all dependencies isolated

## 🛠️ Technology Stack

### System Dependencies
*   **Tesseract OCR:** Multi-language OCR engine (with English and Turkish language packs)
*   **Poppler:** PDF rendering library for pdf2image conversion

### Core Python Libraries
*   **Deep Learning:** PyTorch 2.3.1, TorchVision 0.18.1 (CUDA 12.1)
*   **Transformers:** Hugging Face Transformers 4.41.2, Accelerate 0.31.0
*   **Computer Vision:** OpenCV (opencv-python-headless 4.10.0.84), Pillow 10.3.0
*   **OCR Engines:**
    *   Pytesseract 0.3.10 (Tesseract wrapper)
    *   PaddlePaddle 2.6.1 + PaddleOCR 2.8.1 (Chinese OCR framework)
*   **Document Processing:**
    *   PDF2Image 1.17.0 (PDF to image conversion)
    *   LayoutParser 0.3.4 (Layout detection with PaddleDetection backend)
*   **Scientific Computing:** NumPy 1.26.4, Pandas 2.2.2, scikit-learn 1.5.0
*   **Utilities:** tqdm 4.66.4, langdetect 1.0.9, bitsandbytes 0.43.1 (4-bit quantization)

### Pre-trained AI Models
*   **TrOCR:** `microsoft/trocr-base-handwritten` - Handwriting recognition specialist
*   **ViT:** `google/vit-base-patch16-224` - Figure classification with 1000+ ImageNet classes
*   **Phi-3:** `microsoft/Phi-3-mini-4k-instruct` - LLM for natural language description generation (4-bit quantized)
*   **PaddleOCR:** Latin script model - General purpose OCR with handwriting support
*   **LayoutParser:** PicoDet PaddleDetection model - Document layout detection (5 classes)

### Development Stack
*   **Containerization:** Docker with NVIDIA CUDA 12.1 runtime (nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04)
*   **Python Version:** Python 3.10+
*   **GPU Support:** NVIDIA CUDA-enabled GPU recommended (falls back to CPU with warnings)

## ⚙️ Setup and Getting Started

This project requires Docker and NVIDIA GPU support to run.

### Prerequisites

1.  **Git:** Required to clone the project.
2.  **NVIDIA Drivers:** Ensure you have up-to-date NVIDIA drivers installed on your machine.
3.  **Docker Desktop:** Install it from the [official Docker website](https://www.docker.com/products/docker-desktop/).
4.  **NVIDIA Container Toolkit:** Required for Docker to utilize GPUs. Follow the installation instructions in the [official NVIDIA documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

### Installation Steps

1.  **Clone the Project:**
    ```bash
    git clone https://github.com/SaKinLord/doc_processing_pipeline.git
    cd doc_processing_pipeline
    ```

2.  **Build the Docker Image:**
    Run the following command in the project's root directory. This process might take a while the first time as it downloads the base image and all Python libraries.
    ```bash
    docker build -t doc-ai-pipeline .
    ```

## 🚀 Usage

### Basic Usage with Docker

1.  **Prepare Your Files:** Copy all the PDF and image files you want to process into the `input` folder in the project's root directory.

2.  **Run the Docker Container:**
    The following command will start a container using the `doc-ai-pipeline` image, provide GPU access, and mount your local `input`/`output` folders to the container.

    **Linux / macOS / Windows (WSL):**
    ```bash
    docker run -it --rm --gpus all \
      -v "$(pwd)/input:/app/input" \
      -v "$(pwd)/output:/app/output" \
      doc-ai-pipeline:latest
    ```

    **Windows (PowerShell):**
    ```bash
    docker run -it --rm --gpus all \
      -v "${PWD}/input:/app/input" \
      -v "${PWD}/output:/app/output" \
      doc-ai-pipeline:latest
    ```

3.  **Check the Results:**
    Once the process is complete, the `structured_output.json` file containing all the structured data will be created in the `output` folder. Additionally, detected tables (`.csv`) will be saved in the `output/tables` subfolder.

### Advanced Command-Line Options

The pipeline supports several command-line arguments for customization:

```bash
# Custom output filename
python main.py --output_file my_results.json

# Enable parallel processing for batch operations (file-level)
python main.py --parallel --workers 4

# Enable parallel page processing within PDFs (NEW!)
python main.py --parallel-pages --page-workers 6

# Combine both for maximum performance
python main.py --parallel --workers 4 --parallel-pages --page-workers 8

# Skip model pre-loading for faster startup (models load on-demand)
python main.py --no-preload

# Combine multiple options
python main.py --parallel --parallel-pages --output_file batch_results.json
```

**Using with Docker:**
```bash
docker run -it --rm --gpus all \
  -v "$(pwd)/input:/app/input" \
  -v "$(pwd)/output:/app/output" \
  doc-ai-pipeline:latest \
  python3 main.py --parallel --workers 4
```

### Output Structure

After processing, you'll find the following in the `output` directory:

```
output/
├── structured_output.json    # Main JSON output with all extracted data
├── tables/                    # Extracted tables in CSV format
│   ├── document1_p1_table_lined_1.csv
│   └── document2_p1_table_borderless_1.csv
└── visualizations/           # (Optional) Visual annotations of detected elements
    └── document1_page_1_viz.png
```

### Visualization Tool

To visualize the detected elements on your documents:

```bash
# Run after processing documents
python visualize_output.py

# Or with custom JSON file
python visualize_output.py --json_file output/my_results.json
```

This creates annotated images in `output/visualizations/` showing:
- Text blocks (blue-turquoise boxes)
- Tables (green boxes)
- Figures (red boxes with type labels)
- Form fields (purple labels, yellow values with arrows)

### Understanding the Output JSON

The `structured_output.json` follows this structure:

```json
{
  "documents": [
    {
      "file": "example.pdf",
      "document_type": "pdf",
      "total_pages": 3,
      "pages": [
        {
          "page": 1,
          "page_width": 2480,
          "page_height": 3508,
          "elements": [
            {
              "id": "textblock_1",
              "type": "TextBlock",
              "bounding_box": [x1, y1, x2, y2],
              "content": "Extracted text...",
              "meta": {"ocr_conf": 92.5}
            },
            {
              "id": "table_lined_1",
              "type": "Table",
              "shape": {"rows": 5, "cols": 3},
              "cells": [...],
              "export": {"csv_path": "output/tables/example_p1_table_lined_1.csv"},
              "description": "AI-generated table summary..."
            },
            {
              "id": "figure_1",
              "type": "Figure",
              "bounding_box": [x1, y1, x2, y2],
              "kind": "bar_chart",
              "caption": "Figure 1: Sales data",
              "description": "AI-generated figure description..."
            }
          ],
          "metadata": {
            "language": "en",
            "fields": {
              "date": {
                "value": "2024-01-15",
                "label_bbox": [...],
                "value_bbox": [...],
                "match_direction": "right"
              }
            }
          }
        }
      ]
    }
  ]
}
```

## 📂 Project Structure

```
doc_processing_pipeline/
├── input/                      # Place documents to be processed here
├── output/                     # Output JSON and other files are generated here
│   ├── tables/                 # Extracted tables in CSV format
│   ├── figure_crops/           # (Reserved for future figure crops)
│   └── visualizations/         # Annotated images from visualization tool
│
├── src/                        # Project source code
│   ├── __init__.py
│   ├── config.py               # Configuration constants (paths, thresholds, PSM modes)
│   │
│   ├── pipeline.py             # Main DocumentProcessor orchestration class
│   ├── model_manager.py        # Centralized AI model pre-loading and management
│   │
│   ├── image_utils.py          # Image loading, adaptive preprocessing, quality analysis
│   ├── utils.py                # Helper functions (bbox operations, clustering, IoU)
│   │
│   ├── layout_detection.py    # Layout analysis + graph-based reading order
│   ├── ocr.py                  # Smart OCR router with handwriting detection
│   ├── handwriting_detector.py # Feature-based handwriting vs printed classification (NEW!)
│   │
│   ├── table_extraction.py    # Hybrid table detection (bordered + borderless)
│   ├── table_structure.py      # Advanced table structure with merged cells (NEW!)
│   ├── figure_extraction.py   # Figure detection and caption finding
│   ├── classifier.py           # ViT-based figure classification
│   ├── chart_extractor.py      # Extract numerical data from charts (NEW!)
│   ├── field_extraction.py    # Hybrid rule-based + LLM field extraction
│   │
│   └── description_generator.py # Phi-3 LLM for descriptions and field extraction
│
├── main.py                     # Main execution script with CLI arguments
├── visualize_output.py         # Visualization tool for annotating processed documents
│
├── requirements.txt            # Python dependencies with versions
├── Dockerfile                  # Docker image build instructions (CUDA 12.1)
├── .dockerignore               # Files excluded from Docker build
├── .gitignore                  # Files excluded from Git
└── README.md                   # This file
```

### Module Descriptions

**Core Pipeline:**
- `main.py`: Entry point with argument parsing, directory setup, and batch processing orchestration
- `pipeline.py`: `DocumentProcessor` class that runs the 8-step processing pipeline per page
- `model_manager.py`: Singleton `ModelManager` for centralized model initialization and access

**Image Processing:**
- `image_utils.py`: Image loading, denoising, deskewing, CLAHE, adaptive thresholding, line removal
- `utils.py`: Bounding box utilities, 1D clustering, IoU calculation, text normalization

**Document Understanding:**
- `layout_detection.py`: Layout analysis using LayoutParser (PaddleDetection) or Tesseract PSM 3
- `ocr.py`: Smart 3-stage OCR with fallback chain and confidence thresholding
- `table_extraction.py`: Hough Transform for lined tables, alignment detection for borderless tables
- `figure_extraction.py`: Morphological figure detection and caption association
- `field_extraction.py`: Regex-based form field detection with multi-directional search

**AI Models:**
- `classifier.py`: ViT-based figure classification with 14+ category mapping
- `description_generator.py`: Phi-3 4-bit quantized LLM for table and figure descriptions

**Configuration:**
- `config.py`: Centralized configuration (DPI, thresholds, tolerances, PSM modes, languages)

## 🔧 Configuration

The pipeline can be customized by editing `src/config.py`:

### Image Preprocessing
- `PDF_TO_IMAGE_DPI`: DPI for PDF conversion (default: 300)
- `DENOISING_H`: Denoising strength parameter (default: 7)
- `ADAPTIVE_THRESH_BLOCK_SIZE`: Block size for adaptive thresholding (default: 35)
- `ADAPTIVE_THRESH_C`: Constant for adaptive thresholding (default: 11)
- `DESKEW_HOUGH_THRESHOLD`: Hough Lines threshold for deskew (default: 200)

### Layout Analysis
- `BLOCK_MIN_AREA`: Minimum area for text blocks in pixels (default: 500)
- `MORPH_KERNEL_SIZE`: Morphological kernel size for merging lines (default: (25, 3))
- `COLUMN_REORDER_TOLERANCE`: Tolerance for grouping columns (default: 80)

### OCR Settings
- `DEFAULT_OCR_LANG`: Default Tesseract language (default: "eng+tur")
- `DEFAULT_PSM`: Default Page Segmentation Mode (default: 6)
- `LOW_CONFIDENCE_THRESHOLD`: OCR confidence threshold for fallback (default: 70.0)
- `LANG_DETECT_CONFIDENCE_THRESHOLD`: Language detection confidence (default: 0.98)

### Table Extraction
- `TABLE_HOUGH_MIN_LINE_LEN`: Minimum line length for Hough Transform (default: 80)
- `TABLE_HOUGH_MAX_LINE_GAP`: Maximum line gap (default: 10)
- `TABLE_LINE_MERGE_TOLERANCE`: Tolerance for merging nearby lines (default: 5)
- `TABLE_MIN_CELL_WH`: Minimum cell width/height (default: 14)

### Figure Extraction
- `FIGURE_CAPTION_V_TOLERANCE`: Max vertical distance for caption search (default: 150)

## 🏗️ Processing Pipeline

Each document page goes through an 8-step processing pipeline:

1. **Image Preparation:** Load image → Apply denoising → Deskew → CLAHE → Adaptive threshold
2. **Line Detection & Removal:** Find horizontal/vertical lines → Create line-free image for better text extraction
3. **Language Detection:** Quick OCR scan → Detect language with confidence → Set Tesseract language
4. **Layout Analysis:** LayoutParser/Tesseract → Detect text blocks, tables, figures → Reorder by columns
5. **Table Extraction:**
   - Check for lined tables (Hough Transform)
   - Validate table structure (distinguish from forms)
   - Fall back to borderless detection (alignment-based)
   - Generate AI descriptions
6. **Figure Extraction:**
   - Morphological blob detection
   - Filter overlapping text areas
   - Classify figure type (ViT)
   - Find captions
   - Generate AI descriptions
7. **Field Extraction:** Detect form fields → Match labels with values (multi-directional) → Normalize data
8. **Finalization:** Assemble JSON structure with all extracted data

## 🚨 Troubleshooting

### Model Download Issues
Models are automatically downloaded from Hugging Face on first run. If downloads fail:
- Check internet connection
- Ensure sufficient disk space (~10GB for all models)
- Set Hugging Face cache: `export HF_HOME=/path/to/cache`

### GPU Memory Issues
If you encounter CUDA out-of-memory errors:
- Reduce parallel workers: `--workers 1`
- Process files one at a time (remove `--parallel` flag)
- Increase Docker GPU memory allocation
- Use `--no-preload` to reduce initial memory footprint

### Tesseract Language Packs
To add support for additional languages:
```dockerfile
RUN apt-get update && apt-get install -y \
    tesseract-ocr-deu \
    tesseract-ocr-fra \
    tesseract-ocr-spa
```
Then update `DEFAULT_OCR_LANG` in `config.py`.

### Poor OCR Quality
- Increase PDF_TO_IMAGE_DPI in config.py (try 400 or 600)
- Adjust DENOISING_H for noisy documents (try 10-15)
- Check document language matches DEFAULT_OCR_LANG
- Verify GPU is being used (check console output for CUDA messages)

## ⚠️ Known Limitations

### Current Limitations
*   **LLM Hallucinations:** Significantly reduced via verification layers, though AI-generated *descriptions* (summaries) may still occasionally contain minor inaccuracies. Field extraction is strictly verified.
*   **Rotated Text:** Text at angles other than 0/90/180/270 degrees may not be detected correctly
*   **Handwriting Quality:** TrOCR works best with clear, structured handwriting; highly cursive or messy handwriting may have lower accuracy
*   **Non-Latin Scripts:** PaddleOCR is configured for Latin scripts; support for Asian languages requires model reconfiguration
*   **Chart Data Extraction:** Relative values only; absolute values require OCR of axis labels
*   **Handwriting Detection:** Uses heuristic features; custom training data can improve accuracy

### Performance Considerations
*   **First Run:** Initial startup is slow due to model downloads (~5-10 minutes depending on internet speed)
*   **GPU Required:** CPU-only operation is extremely slow and not recommended for production use
*   **Memory Requirements:** Requires ~8GB VRAM for optimal performance with all models loaded
*   **Parallel Processing:** Limited by GPU memory; 2-4 workers recommended for typical GPUs

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Support for more languages (Arabic, Chinese, Japanese, etc.)
- Fine-tuning layout detection model on specific document types
- Training custom CNN for handwriting detection
- Excel export format for tables
- Improved chart axis label OCR for absolute values
- Optimized GPU memory usage for parallel processing

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## 🙏 Acknowledgments

This project uses the following open-source models and libraries:
- **Tesseract OCR** - Apache License 2.0
- **PaddleOCR** - Apache License 2.0
- **Microsoft TrOCR** - MIT License
- **Microsoft Phi-3** - MIT License
- **Google ViT** - Apache License 2.0
- **LayoutParser** - Apache License 2.0
- **PyTorch** - BSD License
- **Hugging Face Transformers** - Apache License 2.0

Special thanks to the research teams at Microsoft, Google, Baidu, and the open-source community.