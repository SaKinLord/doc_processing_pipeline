"""
Diagnostic to test actual pipeline text detection
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

def test_pipeline_text_detection():
    """Test the full pipeline to see what text is detected."""
    print("="*70)
    print("PIPELINE TEXT DETECTION TEST")
    print("="*70)

    # Import the pipeline
    from src.pipeline import DocumentProcessor
    from src.config import USE_LAYOUTPARSER
    from src.model_manager import ModelManager

    print(f"\n✅ USE_LAYOUTPARSER config: {USE_LAYOUTPARSER}")

    # Initialize ModelManager (required for LayoutParser to work)
    print("\nInitializing ModelManager...")
    ModelManager.initialize(load_all=True)
    print("✅ ModelManager initialized\n")

    # Find test file
    input_dir = Path(__file__).parent / "input"
    if not input_dir.exists():
        print(f"\n❌ Input directory not found: {input_dir}")
        return

    test_files = list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg"))
    if not test_files:
        print(f"\n❌ No test files found in {input_dir}")
        return

    test_file = test_files[0]
    print(f"\n✅ Testing with: {test_file.name}")

    # Process with pipeline
    print("\n" + "-"*70)
    print("RUNNING PIPELINE...")
    print("-"*70)

    try:
        processor = DocumentProcessor(file_path=str(test_file))

        # Run individual steps
        processor._prepare_image()
        processor._detect_and_remove_lines()
        processor._detect_language()
        processor._extract_text_blocks()

        print("\n" + "="*70)
        print("PIPELINE RESULTS")
        print("="*70)

        print(f"\nLayout Table ROIs: {len(processor.layout_table_rois)}")
        for i, roi in enumerate(processor.layout_table_rois):
            print(f"  [{i+1}] Table - bbox: {roi['bounding_box']}, conf: {roi['confidence']:.3f}")

        print(f"\nLayout Figure ROIs: {len(processor.layout_figure_rois)}")
        for i, roi in enumerate(processor.layout_figure_rois):
            print(f"  [{i+1}] Figure - bbox: {roi['bounding_box']}, conf: {roi['confidence']:.3f}")

        print(f"\nText Elements: {len(processor.elements)}")
        for i, elem in enumerate(processor.elements):
            if elem['type'] == 'TextBlock':
                content_preview = elem['content'][:60] + "..." if len(elem['content']) > 60 else elem['content']
                print(f"  [{i+1}] TextBlock - {content_preview}")

        # Diagnosis
        print("\n" + "="*70)
        print("DIAGNOSIS")
        print("="*70)

        if len(processor.elements) == 0:
            print("❌ CRITICAL: No text elements detected!")
            print("   Possible causes:")
            print("   - LayoutParser is classifying everything as Table/Figure")
            print("   - Text regions are being filtered out by NMS")
            print("   - OCR is failing on all text blocks")
            print(f"\n   Debug info:")
            print(f"   - Tables detected: {len(processor.layout_table_rois)}")
            print(f"   - Figures detected: {len(processor.layout_figure_rois)}")
        elif len(processor.elements) < 3:
            print("⚠️  WARNING: Very few text elements detected")
            print(f"   Only {len(processor.elements)} text block(s) found")
            print("   This may indicate text detection issues")
        else:
            print(f"✅ Text detection working: {len(processor.elements)} text blocks found")

    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pipeline_text_detection()
