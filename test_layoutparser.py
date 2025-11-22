"""
Diagnostic script to test LayoutParser model and identify class collapse issues.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Workaround for OpenMP conflict

import sys
import numpy as np
import cv2
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_model_directly():
    """Test the model directly without the pipeline."""
    print("="*70)
    print("LAYOUTPARSER DIAGNOSTIC TEST")
    print("="*70)

    # Import layoutparser
    try:
        import layoutparser as lp
        print("\n✅ LayoutParser imported successfully")
    except ImportError as e:
        print(f"\n❌ Failed to import LayoutParser: {e}")
        return

    # Load the model
    print("\n[1/3] Loading model...")
    try:
        model = lp.models.PaddleDetectionLayoutModel(
            config_path="lp://PubLayNet/ppyolov2_r50vd_dcn_365e/config",
            label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
            enforce_cpu=False,
            extra_config={'threshold': 0.3}
        )
        print("✅ Model loaded successfully")
        print(f"   - Threshold: {model.threshold}")
        print(f"   - Label map: {model.label_map}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Find a test image
    print("\n[2/3] Looking for test images...")
    input_dir = Path(__file__).parent / "input"

    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        print("   Creating a synthetic test image instead...")
        # Create a simple test image with text regions
        test_image = np.ones((800, 600, 3), dtype=np.uint8) * 255
        cv2.rectangle(test_image, (50, 50), (550, 200), (0, 0, 0), 2)
        cv2.putText(test_image, "Title Text", (70, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.rectangle(test_image, (50, 250), (550, 400), (0, 0, 0), 2)
        cv2.putText(test_image, "Body Text", (70, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        test_image_rgb = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    else:
        # Look for PDF or image files
        test_files = list(input_dir.glob("*.pdf")) + list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg"))

        if not test_files:
            print(f"❌ No test files found in {input_dir}")
            return

        test_file = test_files[0]
        print(f"✅ Found test file: {test_file.name}")

        # Load the image
        if test_file.suffix.lower() == '.pdf':
            from pdf2image import convert_from_path
            print("   Converting PDF to image...")
            images = convert_from_path(str(test_file), first_page=1, last_page=1, dpi=200)
            test_image_rgb = np.array(images[0])
        else:
            test_image_bgr = cv2.imread(str(test_file))
            test_image_rgb = cv2.cvtColor(test_image_bgr, cv2.COLOR_BGR2RGB)

        print(f"   Image shape: {test_image_rgb.shape}")

    # Run detection
    print("\n[3/3] Running layout detection...")
    try:
        layout = model.detect(test_image_rgb)
        print(f"✅ Raw detection complete: Found {len(layout)} elements")

        # Manually test NMS
        elements_before_nms = []
        for block in layout:
            x1, y1, x2, y2 = block.coordinates
            elements_before_nms.append({
                "bounding_box": [float(x1), float(y1), float(x2), float(y2)],
                "type": block.type,
                "confidence": float(block.score)
            })

        # Apply NMS inline
        def calc_iou(box1, box2):
            x1_min, y1_min, x1_max, y1_max = box1
            x2_min, y2_min, x2_max, y2_max = box2
            inter_x_min = max(x1_min, x2_min)
            inter_y_min = max(y1_min, y2_min)
            inter_x_max = min(x1_max, x2_max)
            inter_y_max = min(y1_max, y2_max)
            if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
                return 0.0
            inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
            box1_area = (x1_max - x1_min) * (y1_max - y1_min)
            box2_area = (x2_max - x2_min) * (y2_max - y2_min)
            union_area = box1_area + box2_area - inter_area
            return inter_area / union_area if union_area > 0 else 0.0

        sorted_elems = sorted(elements_before_nms, key=lambda x: x['confidence'], reverse=True)
        keep = []
        removed = []
        while sorted_elems:
            current = sorted_elems.pop(0)
            keep.append(current)
            filtered = []
            for elem in sorted_elems:
                iou = calc_iou(current['bounding_box'], elem['bounding_box'])
                if iou < 0.5:
                    filtered.append(elem)
                else:
                    removed.append({'elem': elem, 'suppressed_by': current, 'iou': iou})
            sorted_elems = filtered

        print(f"✅ After NMS filtering: {len(keep)} elements (removed {len(removed)} overlapping)")

        if removed:
            print("\n⚠️  REMOVED ELEMENTS:")
            for r in removed:
                print(f"   - {r['elem']['type']} (conf={r['elem']['confidence']:.3f}) suppressed by")
                print(f"     {r['suppressed_by']['type']} (conf={r['suppressed_by']['confidence']:.3f}, IoU={r['iou']:.3f})")

        if len(layout) == 0:
            print("\n⚠️  WARNING: No elements detected!")
            print("   This could indicate:")
            print("   - The image is too simple/blank")
            print("   - The threshold is too high")
            print("   - The model is not working correctly")
            return

        # Analyze predictions
        print("\n" + "="*70)
        print("DETECTION RESULTS")
        print("="*70)

        class_counts = {}
        for i, block in enumerate(layout):
            element_type = block.type
            score = block.score
            coords = block.coordinates

            # Count classes
            class_counts[element_type] = class_counts.get(element_type, 0) + 1

            print(f"\nElement {i+1}:")
            print(f"   Type: {element_type}")
            print(f"   Confidence: {score:.3f}")
            print(f"   Coordinates: ({coords[0]:.1f}, {coords[1]:.1f}, {coords[2]:.1f}, {coords[3]:.1f})")

        # Show class distribution
        print("\n" + "="*70)
        print("CLASS DISTRIBUTION")
        print("="*70)
        total = len(layout)
        for class_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
            percentage = (count / total) * 100
            print(f"   {class_name:12s}: {count:3d} ({percentage:5.1f}%)")

        # Check for class collapse
        print("\n" + "="*70)
        print("DIAGNOSIS")
        print("="*70)

        if len(class_counts) == 1:
            print("❌ CLASS COLLAPSE DETECTED!")
            print(f"   All {total} elements classified as: {list(class_counts.keys())[0]}")
            print("\n   Possible causes:")
            print("   1. Model weights are corrupted or incorrect")
            print("   2. Label map indices are wrong")
            print("   3. Model architecture mismatch")
            print("   4. PaddleDetection version incompatibility")
            print("\n   RECOMMENDATION: Disable LayoutParser (set USE_LAYOUTPARSER=False in config.py)")
        elif max(class_counts.values()) / total > 0.8:
            dominant_class = max(class_counts.items(), key=lambda x: x[1])[0]
            print(f"⚠️  WARNING: Strong class imbalance detected!")
            print(f"   {dominant_class} dominates with {(max(class_counts.values())/total)*100:.1f}% of predictions")
            print("\n   This may indicate:")
            print("   - Model bias or poor calibration")
            print("   - Threshold issues")
            print("   - Document type mismatch")
        else:
            print("✅ Class distribution looks reasonable")
            print(f"   {len(class_counts)} different classes detected")

    except Exception as e:
        print(f"❌ Detection failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model_directly()
