# Document AI Pipeline (Local & Self-Contained)

This project is a locally runnable, self-contained Document AI pipeline that processes multi-format documents (PDF, scanned images, photos, handwriting, etc.) and converts them into a structured JSON format. The system allows you to securely process your sensitive data on your own infrastructure without the need for external cloud APIs.

## ✨ Key Features

*   **Multi-Format Support:** Processes common document and image formats like PDF, PNG, JPG, and TIFF.
*   **Intelligent Layout Analysis:** Automatically detects text blocks, columns, and the correct reading order in documents.
*   **🧠 3-Stage Hybrid OCR:**
    *   **Tesseract:** For fast and effective printed text recognition.
    *   **PaddleOCR:** Automatically engaged for general-purpose and basic handwriting recognition.
    *   **Microsoft TrOCR:** Used as a specialist model for the most challenging and cursive handwriting.
*   **Automatic Language Detection:** Automatically determines the document's language to enhance OCR accuracy.
*   **Table Extraction:** Detects both bordered and borderless (alignment-based) tables and exports them to CSV format.
*   **AI-Powered Figure Understanding:**
    *   **ViT (Vision Transformer):** Detects and classifies figures within the document such as graphs, photos, maps, and diagrams (e.g., `bar_chart`, `photo`, `plot`).
    *   **Microsoft Phi-3:** Automatically generates natural language captions for detected tables and figures.
*   **Noise Robustness:** Utilizes advanced image preprocessing techniques to handle skewed (deskew), low-quality, and "dirty" documents.
*   **🚀 100% Local and Secure:** All models (including TrOCR, ViT, and Phi-3) run on the local machine. Your data never leaves your premises.
*   **Packaged with Docker:** Delivered as a single, ready-to-run Docker image containing all dependencies.

## 🛠️ Technology Stack

*   **System Dependencies:** Tesseract OCR, Poppler
*   **Python Libraries:** PyTorch, Transformers, OpenCV, PaddleOCR, Pytesseract, LayoutParser, Pillow, PDF2Image.
*   **AI Models:**
    *   `microsoft/trocr-base-handwritten` (Specialist Handwriting OCR)
    *   `google/vit-base-patch16-224` (Figure Classification)
    *   `microsoft/Phi-3-mini-4k-instruct` (Natural Language Caption Generation)
    *   `PaddleOCR` (General OCR & Handwriting)

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
    Once the process is complete, the `structured_output.json` file containing all the structured data will be created in the `output` folder. Additionally, detected tables (`.csv`) and figures (`.png`) will be saved in the `output/tables` and `output/figure_crops` subfolders, respectively.

## 📂 Project Structure

```
doc_processing_pipeline/
├── input/              # Place documents to be processed here
├── output/             # Output JSON and other files are generated here
│   ├── tables/
│   └── figure_crops/
├── src/                # Project source code
│   ├── __init__.py
│   ├── config.py       # Configuration file
│   ├── classifier.py   # Figure classification with ViT
│   ├── ... (other modules)
├── main.py             # Main execution script
├── requirements.txt    # Python dependencies
├── Dockerfile          # Docker image build instructions
├── .gitignore          # Files to be ignored by Git
└── .dockerignore       # Files to be ignored by Docker
```

## ⚠️ Known Limitations

*   Merged cells in tables are not yet fully supported.
*   Captions generated by the LLM may occasionally contain "hallucinations" (plausible but incorrect information).

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
