# visualize_output_improved.py
"""
🔧 IMPROVED Visualization Script for Document AI Pipeline
Fixes:
1. Overlapping labels - smaller font, smarter placement
2. Label clutter - abbreviations, selective labeling
3. Better readability - lower opacity, outline-only option
4. Shows OCR confidence with color coding
5. Filters out obvious false figure detections
"""

import os
import json
import argparse
import cv2
import numpy as np
from pdf2image import convert_from_path
from tqdm import tqdm

# Define project folders
INPUT_DIR = "input"
OUTPUT_DIR = "output"
VISUALIZATIONS_DIR = os.path.join(OUTPUT_DIR, "visualizations")

# Color palette for different element types (in BGR format)
COLORS = {
    "TextBlock": (255, 150, 50),      # Blue-ish
    "TextBlock_LowConf": (100, 100, 255),  # Red-ish for low confidence
    "Table": (50, 255, 50),           # Green
    "Figure": (50, 50, 255),          # Red
    "Field_Label": (255, 50, 255),    # Purple/Magenta
    "Field_Value": (255, 255, 50),    # Cyan
}
DEFAULT_COLOR = (128, 128, 128)  # Gray

# 🔧 FIX: Abbreviated labels to reduce clutter
LABEL_ABBREVIATIONS = {
    "TextBlock": "T",
    "Table": "Tbl",
    "Figure": "Fig",
}


def is_likely_false_figure(box, page_width, page_height, min_content_ratio=0.1):
    """
    🔧 FIX: Filter out giant figure boxes that are likely false detections.
    
    Returns True if the figure is likely a false detection (should be skipped).
    Criteria:
    - Covers more than 25% of the page
    - Is mostly in the lower half (likely signature area misdetection)
    """
    x1, y1, x2, y2 = box
    box_width = x2 - x1
    box_height = y2 - y1
    box_area = box_width * box_height
    page_area = page_width * page_height
    
    coverage_ratio = box_area / page_area if page_area > 0 else 0
    
    # Check if box is in lower portion of page
    box_center_y = (y1 + y2) / 2
    is_lower_half = box_center_y > page_height * 0.4
    
    # Filter criteria:
    # 1. Covers more than 25% of the page
    # 2. OR covers more than 15% AND is in lower half (likely signature area)
    if coverage_ratio > 0.25:
        return True
    if coverage_ratio > 0.15 and is_lower_half:
        return True
    
    return False


def draw_box_with_label(image, box, label, color, alpha=0.3, show_label=True, 
                        font_scale=0.35, outline_only=False):
    """
    🔧 IMPROVED: Draws a labeled box with better visibility options.
    
    Changes from original:
    - Lower default alpha (0.3 vs 0.6)
    - Smaller font (0.35 vs 0.7)
    - Option for outline-only mode
    - Smarter label positioning
    """
    x1, y1, x2, y2 = map(int, box)
    
    if outline_only:
        # Just draw the outline, no fill
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    else:
        # Create a semi-transparent layer for the box
        overlay = image.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        # Draw the outer frame
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
    
    if show_label and label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_thickness = 1
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, font_thickness
        )
        
        # Position label above box, but keep it on screen
        label_y = y1 - 3
        if label_y < text_height + 5:
            label_y = y1 + text_height + 3  # Put below if no room above
        
        label_x = x1
        
        # Draw small background for label
        padding = 2
        cv2.rectangle(
            image,
            (label_x - padding, label_y - text_height - padding),
            (label_x + text_width + padding, label_y + padding),
            color,
            -1
        )
        cv2.putText(image, label, (label_x, label_y), font, font_scale, 
                    (255, 255, 255), font_thickness)


def get_abbreviated_label(elem_type, element, use_abbreviations=True):
    """Get a shorter label for the element type."""
    if use_abbreviations:
        base_label = LABEL_ABBREVIATIONS.get(elem_type, elem_type[:3])
    else:
        base_label = elem_type
    
    # Add figure type if available
    if elem_type == "Figure" and "figure_type" in element:
        fig_type = element.get("figure_type", "")[:3]
        return f"{base_label}:{fig_type}"
    
    return base_label


def is_likely_noise_artifact(element, page_width, page_height, edge_margin_percent=0.05):
    """
    🔧 FIX: Determines if a text block is likely noise/artifact rather than real text.
    Filters out scanning artifacts, page edge noise, and OCR garbage.
    """
    if 'bounding_box' not in element:
        return False
    
    x1, y1, x2, y2 = element['bounding_box']
    box_width = x2 - x1
    box_height = y2 - y1
    box_area = box_width * box_height
    page_area = page_width * page_height
    
    # Calculate edge margins
    left_margin = page_width * edge_margin_percent
    right_margin = page_width * (1 - edge_margin_percent)
    top_margin = page_height * edge_margin_percent
    bottom_margin = page_height * (1 - edge_margin_percent)
    
    # Check if near edge
    near_left = x1 < left_margin
    near_right = x2 > right_margin
    near_top = y1 < top_margin
    near_bottom = y2 > bottom_margin
    is_at_edge = near_left or near_right or near_top or near_bottom
    
    # Get content and confidence
    content = element.get('content', '').strip()
    meta = element.get('meta', {})
    ocr_conf = meta.get('ocr_conf', 50)
    
    # Calculate size ratio
    size_ratio = box_area / page_area if page_area > 0 else 0
    
    # RULE 1: Very small boxes at edges with low confidence
    if size_ratio < 0.002 and is_at_edge and ocr_conf < 30:
        return True
    
    # RULE 2: Empty or near-empty content
    if len(content) < 2:
        return True
    
    # RULE 3: Content is mostly non-alphanumeric (likely OCR garbage)
    alphanumeric_count = sum(1 for c in content if c.isalnum())
    if len(content) > 0 and alphanumeric_count / len(content) < 0.3:
        return True
    
    # RULE 4: Very thin boxes at page edges (scanning artifacts)
    aspect_ratio = box_width / box_height if box_height > 0 else 0
    if is_at_edge:
        if aspect_ratio > 20 and box_height < 20:
            return True
        if aspect_ratio < 0.05 and box_width < 20:
            return True
    
    # RULE 5: Boxes at bottom edge with very low confidence
    if near_bottom and ocr_conf < 25 and size_ratio < 0.01:
        return True
    
    # RULE 6: Full-width boxes at very bottom (page footer artifacts)
    if box_width > page_width * 0.8 and y1 > page_height * 0.95:
        return True
    
    return False


def main():
    parser = argparse.ArgumentParser(description="Visualize Document AI Pipeline Output (Improved)")
    parser.add_argument(
        "--json_file",
        type=str,
        default=os.path.join(OUTPUT_DIR, "structured_output.json"),
        help="Path to the structured_output.json file."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "minimal", "outline", "fields_only"],
        default="minimal",
        help="Visualization mode: full (all labels), minimal (abbreviated), outline (no fill), fields_only"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.25,
        help="Transparency of box fills (0.0-1.0). Lower = more transparent."
    )
    parser.add_argument(
        "--font_scale",
        type=float,
        default=0.35,
        help="Font size for labels (default 0.35, original was 0.7)"
    )
    parser.add_argument(
        "--skip_false_figures",
        action="store_true",
        default=True,
        help="Skip figure boxes that are likely false detections"
    )
    parser.add_argument(
        "--show_confidence",
        action="store_true",
        default=True,
        help="Color-code text blocks by OCR confidence"
    )
    parser.add_argument(
        "--max_labels",
        type=int,
        default=30,
        help="Maximum number of TextBlock labels to show (reduces clutter)"
    )
    parser.add_argument(
        "--filter_noise",
        action="store_true",
        default=True,
        help="Filter out noise/artifact text blocks at page edges"
    )
    args = parser.parse_args()

    os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)

    try:
        with open(args.json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: '{args.json_file}' not found. Please run main.py first.")
        return

    print(f"📄 Reading {len(data['documents'])} documents from JSON file...")
    print(f"🎨 Visualization mode: {args.mode}")
    print(f"   - Alpha: {args.alpha}")
    print(f"   - Font scale: {args.font_scale}")
    print(f"   - Skip false figures: {args.skip_false_figures}")
    print(f"   - Filter noise: {args.filter_noise}")
    print(f"   - Max labels: {args.max_labels}")

    for doc in tqdm(data['documents'], desc="Visualizing Documents"):
        original_file_path = os.path.join(INPUT_DIR, doc['file'])
        if not os.path.exists(original_file_path):
            print(f"⚠️ Warning: Original file '{doc['file']}' not found. Skipping.")
            continue

        images = []
        if doc['document_type'] == 'pdf':
            pil_images = convert_from_path(original_file_path, dpi=200)
            for pil_image in pil_images:
                images.append(cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR))
        else:
            images.append(cv2.imread(original_file_path))

        for page_index, page_data in enumerate(doc['pages']):
            if page_index >= len(images):
                continue
            
            page_image = images[page_index]
            page_height, page_width = page_image.shape[:2]
            page_num = page_data['page']
            
            # Count elements for smart labeling
            text_block_count = 0
            noise_filtered_count = 0
            
            # Determine visualization settings based on mode
            use_abbreviations = args.mode in ["minimal", "outline"]
            outline_only = args.mode == "outline"
            show_text_labels = args.mode != "fields_only"
            
            # --- ELEMENT DRAWING LOOP ---
            for element in page_data.get('elements', []):
                elem_type = element.get('type')
                if not elem_type:
                    continue
                
                # 🔧 FIX: Filter noise artifacts
                if args.filter_noise and elem_type == 'TextBlock':
                    if is_likely_noise_artifact(element, page_width, page_height):
                        noise_filtered_count += 1
                        continue

                box = None
                label = get_abbreviated_label(elem_type, element, use_abbreviations)
                
                # Get base color
                color = COLORS.get(elem_type, DEFAULT_COLOR)

                if elem_type == 'Table':
                    if 'cells' in element and element['cells']:
                        all_bboxes = [cell['bbox'] for cell in element['cells'] if 'bbox' in cell]
                        if all_bboxes:
                            min_x1 = min(b[0] for b in all_bboxes)
                            min_y1 = min(b[1] for b in all_bboxes)
                            max_x2 = max(b[2] for b in all_bboxes)
                            max_y2 = max(b[3] for b in all_bboxes)
                            box = [min_x1, min_y1, max_x2, max_y2]
                            
                elif elem_type == 'Figure':
                    if 'bounding_box' in element:
                        box = element['bounding_box']
                        
                        # 🔧 FIX: Skip likely false figure detections
                        if args.skip_false_figures and is_likely_false_figure(
                            box, page_width, page_height
                        ):
                            continue  # Skip this figure
                        
                        if "figure_type" in element:
                            fig_type = element.get('figure_type', 'unknown')
                            label = f"Fig:{fig_type[:4]}" if use_abbreviations else f"Figure ({fig_type})"
                            
                elif elem_type == 'TextBlock':
                    if 'bounding_box' in element:
                        box = element['bounding_box']
                        text_block_count += 1
                        
                        # 🔧 FIX: Color-code by confidence
                        if args.show_confidence:
                            conf = element.get('meta', {}).get('ocr_conf', 50)
                            if conf < 40:
                                color = COLORS.get("TextBlock_LowConf", (100, 100, 255))
                        
                        # 🔧 FIX: Limit label count to reduce clutter
                        show_this_label = show_text_labels and text_block_count <= args.max_labels
                        
                        if box:
                            draw_box_with_label(
                                page_image, box, label if show_this_label else "",
                                color, alpha=args.alpha,
                                show_label=show_this_label,
                                font_scale=args.font_scale,
                                outline_only=outline_only
                            )
                        continue  # Already drew, skip the common draw below
                        
                else:
                    if 'bounding_box' in element:
                        box = element['bounding_box']
                
                # Draw the box if valid
                if box:
                    draw_box_with_label(
                        page_image, box, label,
                        color, alpha=args.alpha,
                        show_label=True,
                        font_scale=args.font_scale,
                        outline_only=outline_only
                    )
            
            # --- FORM FIELDS DRAWING ---
            for field_name, field_data in page_data.get('metadata', {}).get('fields', {}).items():
                # Skip fields without proper bounding boxes
                if 'label_bbox' not in field_data and 'value_bbox' not in field_data:
                    continue
                    
                if 'label_bbox' in field_data:
                    label_box = field_data['label_bbox']
                    field_label = f"F:{field_name[:4]}(L)" if use_abbreviations else f"Field: {field_name} (Label)"
                    draw_box_with_label(
                        page_image, label_box, field_label,
                        COLORS['Field_Label'], alpha=args.alpha + 0.1,
                        font_scale=args.font_scale,
                        outline_only=outline_only
                    )
                
                if 'value_bbox' in field_data:
                    value_box = field_data['value_bbox']
                    field_label = f"F:{field_name[:4]}(V)" if use_abbreviations else f"Field: {field_name} (Value)"
                    draw_box_with_label(
                        page_image, value_box, field_label,
                        COLORS['Field_Value'], alpha=args.alpha + 0.1,
                        font_scale=args.font_scale,
                        outline_only=outline_only
                    )

                # Draw arrow connecting label to value
                if 'label_bbox' in field_data and 'value_bbox' in field_data:
                    label_box = field_data['label_bbox']
                    value_box = field_data['value_bbox']
                    lx = int((label_box[0] + label_box[2]) / 2)
                    ly = int((label_box[1] + label_box[3]) / 2)
                    vx = int((value_box[0] + value_box[2]) / 2)
                    vy = int((value_box[1] + value_box[3]) / 2)
                    cv2.arrowedLine(page_image, (lx, ly), (vx, vy), (0, 255, 255), 1, tipLength=0.05)

            # Save output
            base_name = os.path.splitext(doc['file'])[0]
            output_filename = f"{base_name}_page_{page_num}_viz.png"
            output_path = os.path.join(VISUALIZATIONS_DIR, output_filename)
            cv2.imwrite(output_path, page_image)

    print(f"\n✅ Visualization completed! Outputs saved to '{VISUALIZATIONS_DIR}' folder.")
    print("\n💡 Tips:")
    print("   - Use --mode=outline for cleaner view of dense documents")
    print("   - Use --mode=fields_only to see only form field detections")
    print("   - Use --alpha=0.15 for even more transparency")
    print("   - Use --max_labels=10 to reduce label clutter on busy documents")


if __name__ == "__main__":
    main()
