# main.py

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import glob
import json
import argparse
import time
from tqdm import tqdm
from pdf2image import convert_from_path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from src.pipeline import DocumentProcessor
from src.config import INPUT_DIR, OUTPUT_DIR, TABLES_DIR, FIGURES_DIR, PDF_TO_IMAGE_DPI, DEFAULT_OCR_LANG
from src.model_manager import ModelManager

def setup_directories():
    """Creates necessary input and output directories."""
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print(f"Input directory: {os.path.abspath(INPUT_DIR)}")
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")

def find_input_files():
    """Finds all files with supported formats in the input directory."""
    supported_formats = ['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'pdf']
    files = set()
    for fmt in supported_formats:
        files.update(glob.glob(os.path.join(INPUT_DIR, f'*.{fmt}')))
        files.update(glob.glob(os.path.join(INPUT_DIR, f'*.{fmt.upper()}')))
    return list(files)


def process_single_image(file_path):
    """
    Processes a single image file. Used for parallel processing.
    Returns the document data or None if processing failed.
    """
    try:
        file_name = os.path.basename(file_path)
        processor = DocumentProcessor(file_path=file_path)
        result = processor.process()
        if result:
            return {
                "file": file_name,
                "document_type": "image",
                "pages": [result]
            }
    except Exception as e:
        print(f"\n[ERROR] Failed to process image {os.path.basename(file_path)}: {e}")
    return None


def process_single_pdf(file_path):
    """
    Processes a single PDF file. Used for parallel processing.
    Returns the document data or None if processing failed.
    """
    try:
        file_name = os.path.basename(file_path)
        print(f"\nProcessing PDF: {file_name}")
        images = convert_from_path(file_path, dpi=PDF_TO_IMAGE_DPI)
        pdf_pages_data = []
        
        document_language = None
        
        for i, image in enumerate(tqdm(images, desc=f"  Pages of {file_name}", leave=False)):
            page_num = i + 1
            
            lang_override = document_language if document_language else None
            
            processor = DocumentProcessor(
                image_object=image, 
                file_name=file_name, 
                page_num=page_num,
                language_override=lang_override
            )
            result = processor.process()
            
            if document_language is None:
                if processor.detected_lang_tess == DEFAULT_OCR_LANG:
                    document_language = DEFAULT_OCR_LANG
                else:
                    document_language = processor.detected_lang_tess

            if result:
                pdf_pages_data.append(result)
        
        if pdf_pages_data:
            return {
                "file": file_name,
                "document_type": "pdf",
                "total_pages": len(pdf_pages_data),
                "pages": pdf_pages_data
            }
    except Exception as e:
        print(f"\n[ERROR] Failed to process PDF {os.path.basename(file_path)}: {e}")
    return None


def main():
    """Main program function."""
    start_time = time.time()
    
    parser = argparse.ArgumentParser(description="Document Processing Pipeline")
    parser.add_argument(
        "--output_file", 
        type=str, 
        default="structured_output.json",
        help="Name of the final JSON output file."
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel processing of multiple files (faster for batches)."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count / 2)."
    )
    parser.add_argument(
        "--no-preload",
        action="store_true",
        help="Skip pre-loading AI models (may be slower per page but faster startup)."
    )
    args = parser.parse_args()

    setup_directories()
    
    # Pre-load all AI models (unless --no-preload is specified)
    if not args.no_preload:
        ModelManager.initialize(load_all=True)
    else:
        print("\n⊘ Skipping model pre-loading (models will load on-demand)")
    
    input_files = find_input_files()
    if not input_files:
        print("\nNo supported files (images or PDFs) found in the input directory. Please add some files and run again.")
        return

    print(f"\nFound {len(input_files)} files to process.")
    
    # Separate PDFs and images
    pdf_files = [f for f in input_files if f.lower().endswith('.pdf')]
    image_files = [f for f in input_files if not f.lower().endswith('.pdf')]
    
    all_documents_data = []
    
    # Process based on parallel flag
    if args.parallel and len(input_files) > 1:
        # Parallel processing
        num_workers = args.workers if args.workers else max(1, multiprocessing.cpu_count() // 2)
        print(f"\n🚀 Using parallel processing with {num_workers} workers")
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_file = {}
            
            # Submit image tasks
            for file_path in image_files:
                future = executor.submit(process_single_image, file_path)
                future_to_file[future] = file_path
            
            # Note: PDFs are processed sequentially due to page-by-page processing complexity
            # But multiple PDFs can be processed in parallel
            for file_path in pdf_files:
                future = executor.submit(process_single_pdf, file_path)
                future_to_file[future] = file_path
            
            # Collect results
            for future in tqdm(as_completed(future_to_file), total=len(future_to_file), desc="Processing Files"):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    if result:
                        all_documents_data.append(result)
                except Exception as e:
                    print(f"\n[ERROR] Failed to process {os.path.basename(file_path)}: {e}")
    
    else:
        # Sequential processing (original behavior)
        print("\n📄 Using sequential processing")
        
        for file_path in tqdm(input_files, desc="Processing Files"):
            file_name = os.path.basename(file_path)
            _, file_extension = os.path.splitext(file_name)

            if file_extension.lower() == '.pdf':
                result = process_single_pdf(file_path)
                if result:
                    all_documents_data.append(result)
            else:
                result = process_single_image(file_path)
                if result:
                    all_documents_data.append(result)

    output_path = os.path.join(OUTPUT_DIR, args.output_file)
    final_output = {"documents": all_documents_data}
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Processing complete! Output saved to: {os.path.abspath(output_path)}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("--------------------------------------------------")
    print(f"Total Execution Time: {elapsed_time:.2f} seconds")
    print(f"Which is approximately {elapsed_time / 60:.2f} minutes.")
    print("--------------------------------------------------")


if __name__ == "__main__":
    main()