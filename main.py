# main.py

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import glob
import json
import argparse
import time
from tqdm import tqdm
from pdf2image import convert_from_path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
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


def process_single_page(args):
    """
    Processes a single PDF page. Used for parallel page processing.

    Args:
        args: Tuple of (image, file_name, page_num, language_override)

    Returns:
        Tuple of (page_num, result, detected_language) or (page_num, None, None)
    """
    image, file_name, page_num, language_override = args
    try:
        processor = DocumentProcessor(
            image_object=image,
            file_name=file_name,
            page_num=page_num,
            language_override=language_override
        )
        result = processor.process()
        detected_lang = processor.detected_lang_tess
        return (page_num, result, detected_lang)
    except Exception as e:
        print(f"\n[ERROR] Failed to process page {page_num} of {file_name}: {e}")
        return (page_num, None, None)


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


def process_single_pdf(file_path, parallel_pages=False, max_page_workers=4):
    """
    Processes a single PDF file.

    Args:
        file_path: Path to the PDF file
        parallel_pages: If True, process pages in parallel using ThreadPoolExecutor
        max_page_workers: Maximum number of parallel page workers

    Returns:
        Document data or None if processing failed.
    """
    try:
        file_name = os.path.basename(file_path)
        print(f"\nProcessing PDF: {file_name}")
        images = convert_from_path(file_path, dpi=PDF_TO_IMAGE_DPI)

        if len(images) == 0:
            print(f"    - No pages found in PDF")
            return None

        pdf_pages_data = []

        if parallel_pages and len(images) > 1:
            # Parallel page processing using ThreadPoolExecutor
            # We use threads instead of processes to share GPU resources
            print(f"    - Using parallel page processing with {min(max_page_workers, len(images))} workers")

            # First, process page 1 to detect language
            first_page_result = process_single_page((images[0], file_name, 1, None))
            _, first_result, document_language = first_page_result

            if first_result:
                pdf_pages_data.append(first_result)
            else:
                document_language = DEFAULT_OCR_LANG

            # Process remaining pages in parallel with detected language
            if len(images) > 1:
                remaining_tasks = [
                    (images[i], file_name, i + 1, document_language)
                    for i in range(1, len(images))
                ]

                with ThreadPoolExecutor(max_workers=max_page_workers) as executor:
                    futures = {
                        executor.submit(process_single_page, task): task[2]
                        for task in remaining_tasks
                    }

                    results = {}
                    for future in tqdm(
                        as_completed(futures),
                        total=len(futures),
                        desc=f"  Pages 2-{len(images)} of {file_name}",
                        leave=False
                    ):
                        page_num = futures[future]
                        try:
                            p_num, result, _ = future.result()
                            if result:
                                results[p_num] = result
                        except Exception as e:
                            print(f"\n[ERROR] Page {page_num} failed: {e}")

                    # Sort results by page number and append
                    for page_num in sorted(results.keys()):
                        pdf_pages_data.append(results[page_num])
        else:
            # Sequential processing (original behavior)
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
        "--parallel-pages",
        action="store_true",
        help="Enable parallel processing of pages within PDFs (faster for large PDFs)."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers for file processing (default: CPU count / 2)."
    )
    parser.add_argument(
        "--page-workers",
        type=int,
        default=4,
        help="Number of parallel workers for page processing within PDFs (default: 4)."
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
        if getattr(args, 'parallel_pages', False):
            print(f"    + Parallel page processing enabled with {args.page_workers} page workers")

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
                future = executor.submit(
                    process_single_pdf,
                    file_path,
                    getattr(args, 'parallel_pages', False),
                    args.page_workers
                )
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
        if getattr(args, 'parallel_pages', False):
            print(f"    + Parallel page processing enabled with {args.page_workers} page workers")

        for file_path in tqdm(input_files, desc="Processing Files"):
            file_name = os.path.basename(file_path)
            _, file_extension = os.path.splitext(file_name)

            if file_extension.lower() == '.pdf':
                result = process_single_pdf(
                    file_path,
                    getattr(args, 'parallel_pages', False),
                    args.page_workers
                )
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