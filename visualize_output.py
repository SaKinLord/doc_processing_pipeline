# visualize_output.py

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
    "TextBlock": (255, 150, 50),     # Blue-Turquoise
    "Table": (50, 255, 50),          # Green
    "Figure": (50, 50, 255),         # Red
    "Field_Label": (255, 50, 255),    # Purple
    "Field_Value": (255, 255, 50),    # Light Blue
}
DEFAULT_COLOR = (128, 128, 128) # Gray

def draw_box_with_label(image, box, label, color, alpha=0.6):
    """Draws a labeled and semi-transparent box on an image."""
    x1, y1, x2, y2 = map(int, box)
    
    # Create a semi-transparent layer for the box
    overlay = image.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    
    # Merge the original image with the layer
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    
    # Draw the outer frame of the box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    
    # Place the label text above the box
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    
    # Draw a background for the label by getting text size
    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
    label_y = y1 - 10 if y1 - 10 > 10 else y1 + text_height + 10
    
    cv2.rectangle(image, (x1, label_y - text_height - 5), (x1 + text_width, label_y + 5), color, -1)
    cv2.putText(image, label, (x1, label_y), font, font_scale, (255, 255, 255), font_thickness)

def main():
    parser = argparse.ArgumentParser(description="Visualize Document AI Pipeline Output")
    parser.add_argument(
        "--json_file",
        type=str,
        default=os.path.join(OUTPUT_DIR, "structured_output.json"),
        help="Path to the structured_output.json file."
    )
    args = parser.parse_args()

    os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)

    try:
        with open(args.json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: '{args.json_file}' not found. Please run main.py first.")
        return

    print(f"Reading {len(data['documents'])} documents from JSON file...")

    for doc in tqdm(data['documents'], desc="Visualizing Documents"):
        original_file_path = os.path.join(INPUT_DIR, doc['file'])
        if not os.path.exists(original_file_path):
            print(f"Warning: Original file '{doc['file']}' not found in input folder. Skipping.")
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
            page_num = page_data['page']

            # --- CORRECTED ELEMENT DRAWING LOOP ---
            for element in page_data.get('elements', []):
                elem_type = element.get('type')
                if not elem_type:
                    continue

                color = COLORS.get(elem_type, DEFAULT_COLOR)
                box = None
                label = elem_type

                if elem_type == 'Table':
                    # If element is a table, calculate a general box covering all its cells
                    if 'cells' in element and element['cells']:
                        all_bboxes = [cell['bbox'] for cell in element['cells'] if 'bbox' in cell]
                        if all_bboxes:
                            min_x1 = min(b[0] for b in all_bboxes)
                            min_y1 = min(b[1] for b in all_bboxes)
                            max_x2 = max(b[2] for b in all_bboxes)
                            max_y2 = max(b[3] for b in all_bboxes)
                            box = [min_x1, min_y1, max_x2, max_y2]
                elif 'bounding_box' in element:
                    # For other elements, use bounding_box directly
                    box = element['bounding_box']
                    if elem_type == "Figure" and "kind" in element:
                        label = f"Figure ({element.get('kind', 'unknown')})"
                
                # If a valid box is found, draw it
                if box:
                    draw_box_with_label(page_image, box, label, color)
            
            # Form fields drawing logic (no changes)
            for field_name, field_data in page_data.get('metadata', {}).get('fields', {}).items():
                if 'label_bbox' in field_data and 'value_bbox' in field_data:
                    label_box = field_data['label_bbox']
                    draw_box_with_label(page_image, label_box, f"Field: {field_name} (Label)", COLORS['Field_Label'])
                    
                    value_box = field_data['value_bbox']
                    draw_box_with_label(page_image, value_box, f"Field: {field_name} (Value)", COLORS['Field_Value'])

                    lx, ly = int((label_box[0] + label_box[2]) / 2), int((label_box[1] + label_box[3]) / 2)
                    vx, vy = int((value_box[0] + value_box[2]) / 2), int((value_box[1] + value_box[3]) / 2)
                    cv2.arrowedLine(page_image, (lx, ly), (vx, vy), (0, 255, 255), 2, tipLength=0.03)

            base_name = os.path.splitext(doc['file'])[0]
            output_filename = f"{base_name}_page_{page_num}_viz.png"
            output_path = os.path.join(VISUALIZATIONS_DIR, output_filename)
            cv2.imwrite(output_path, page_image)

    print(f"\n✅ Visualization completed! Outputs saved to '{VISUALIZATIONS_DIR}' folder.")

if __name__ == "__main__":
    main()