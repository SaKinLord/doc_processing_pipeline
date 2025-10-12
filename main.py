# main.py

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import glob
import json
import argparse
import time  # <-- 1. Added module for time measurement
from tqdm import tqdm
from pdf2image import convert_from_path
from src.pipeline import DocumentProcessor
from src.config import INPUT_DIR, OUTPUT_DIR, TABLES_DIR, FIGURES_DIR, PDF_TO_IMAGE_DPI, DEFAULT_OCR_LANG

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


def main():
    """Main program function."""
    start_time = time.time()  # <-- 2. Recorded the program's start time
    
    parser = argparse.ArgumentParser(description="Document Processing Pipeline")
    parser.add_argument(
        "--output_file", 
        type=str, 
        default="structured_output.json",
        help="Name of the final JSON output file."
    )
    args = parser.parse_args()

    setup_directories()
    
    input_files = find_input_files()
    if not input_files:
        print("\nNo supported files (images or PDFs) found in the input directory. Please add some files and run again.")
        return

    print(f"\nFound {len(input_files)} files to process.")
    
    all_documents_data = []
    
    for file_path in tqdm(input_files, desc="Processing Files"):
        file_name = os.path.basename(file_path)
        _, file_extension = os.path.splitext(file_name)

        if file_extension.lower() == '.pdf':
            try:
                print(f"\nProcessing PDF: {file_name}")
                images = convert_from_path(file_path, dpi=PDF_TO_IMAGE_DPI)
                pdf_pages_data = []
                
                document_language = None
                
                for i, image in enumerate(tqdm(images, desc=f"  Pages of {file_name}", leave=False)):
                    page_num = i + 1
                    
                    # UPDATED: Improved language logic
                    # If the document's language hasn't been determined yet (first page), detect it.
                    # For subsequent pages, use the determined language.
                    lang_override = document_language if document_language else None
                    
                    processor = DocumentProcessor(
                        image_object=image, 
                        file_name=file_name, 
                        page_num=page_num,
                        language_override=lang_override
                    )
                    result = processor.process()
                    
                    # After processing the first page, fix the document's language
                    if document_language is None:
                        # If language detection fell back to the default language,
                        # use the default for the entire document.
                        if processor.detected_lang_tess == DEFAULT_OCR_LANG:
                             document_language = DEFAULT_OCR_LANG
                             print(f"    - Setting document language to default: '{DEFAULT_OCR_LANG}'")
                        # Otherwise, use the confidently detected language.
                        else:
                             document_language = processor.detected_lang_tess
                             print(f"    - Setting document language to: '{document_language}'")

                    if result:
                        pdf_pages_data.append(result)
                
                if pdf_pages_data:
                    all_documents_data.append({
                        "file": file_name,
                        "document_type": "pdf",
                        "total_pages": len(pdf_pages_data),
                        "pages": pdf_pages_data
                    })

            except Exception as e:
                print(f"\n[ERROR] Failed to process PDF {file_name}: {e}")

        else: # It's an image file
            print(f"\nProcessing Image: {file_name}")
            processor = DocumentProcessor(file_path=file_path)
            result = processor.process()
            if result:
                all_documents_data.append({
                    "file": file_name,
                    "document_type": "image",
                    "pages": [result]
                })

    output_path = os.path.join(OUTPUT_DIR, args.output_file)
    final_output = {"documents": all_documents_data}
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Processing complete! Output saved to: {os.path.abspath(output_path)}")

    # <-- 3. Calculated and printed the elapsed time at the end of the program
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("--------------------------------------------------")
    print(f"Total Execution Time: {elapsed_time:.2f} seconds")
    print(f"Which is approximately {elapsed_time / 60:.2f} minutes.")
    print("--------------------------------------------------")


if __name__ == "__main__":
    main()