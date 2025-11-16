# src/table_structure.py

"""
Table Structure Recognition Module

Handles complex table structures including:
- Merged cells (rowspan/colspan)
- Irregular table layouts
- HTML/Markdown export with proper structure
"""

import cv2
import numpy as np
import pandas as pd
from . import utils


class TableStructureAnalyzer:
    """
    Analyzes table structure to detect merged cells and complex layouts.
    """

    def __init__(self):
        self.grid = None
        self.num_rows = 0
        self.num_cols = 0
        self.cell_map = {}  # (row, col) -> cell info

    def analyze_from_lines(self, h_lines, v_lines, image_shape):
        """
        Analyzes table structure from detected lines.

        Args:
            h_lines: List of horizontal lines [(x1, y1, x2, y2), ...]
            v_lines: List of vertical lines [(x1, y1, x2, y2), ...]
            image_shape: (height, width) of the image

        Returns:
            dict with structure info including merged cells
        """
        if len(h_lines) < 2 or len(v_lines) < 2:
            return None

        # Get unique y-coordinates from horizontal lines (row boundaries)
        y_coords = sorted(set(y for _, y, _, _ in h_lines))

        # Get unique x-coordinates from vertical lines (column boundaries)
        x_coords = sorted(set(x for x, _, _, _ in v_lines))

        self.num_rows = len(y_coords) - 1
        self.num_cols = len(x_coords) - 1

        if self.num_rows < 1 or self.num_cols < 1:
            return None

        # Create base grid
        self.grid = [[None for _ in range(self.num_cols)] for _ in range(self.num_rows)]

        # Detect merged cells by checking line continuity
        merged_cells = self._detect_merged_cells(h_lines, v_lines, x_coords, y_coords)

        # Build cell map
        self._build_cell_map(x_coords, y_coords, merged_cells)

        return {
            'num_rows': self.num_rows,
            'num_cols': self.num_cols,
            'x_coords': x_coords,
            'y_coords': y_coords,
            'cell_map': self.cell_map,
            'merged_cells': merged_cells
        }

    def _detect_merged_cells(self, h_lines, v_lines, x_coords, y_coords):
        """
        Detects merged cells by checking for missing internal lines.

        A cell is merged if:
        - Horizontal line doesn't span the full cell width (vertical merge)
        - Vertical line doesn't span the full cell height (horizontal merge)
        """
        merged_cells = []

        # Build line presence maps
        h_line_map = self._build_line_map(h_lines, axis='h', coords=y_coords)
        v_line_map = self._build_line_map(v_lines, axis='v', coords=x_coords)

        # Track which cells are already part of a merge
        processed = set()

        # Scan for merged regions
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                if (row, col) in processed:
                    continue

                # Check how far this cell extends
                rowspan = 1
                colspan = 1

                # Check for vertical merge (rowspan)
                for r in range(row + 1, self.num_rows):
                    # Check if horizontal line exists between row r-1 and r at this column
                    if self._has_horizontal_separator(h_line_map, r, col, x_coords, y_coords):
                        break
                    rowspan += 1

                # Check for horizontal merge (colspan)
                for c in range(col + 1, self.num_cols):
                    # Check if vertical line exists between col c-1 and c at this row
                    if self._has_vertical_separator(v_line_map, row, c, x_coords, y_coords):
                        break
                    colspan += 1

                # Verify the merged region is consistent
                if rowspan > 1 or colspan > 1:
                    # Verify all internal separators are missing
                    is_valid_merge = True

                    for r in range(row, row + rowspan):
                        for c in range(col, col + colspan):
                            # Check internal horizontal lines
                            if r < row + rowspan - 1:
                                if self._has_horizontal_separator(h_line_map, r + 1, c, x_coords, y_coords):
                                    is_valid_merge = False
                                    break
                            # Check internal vertical lines
                            if c < col + colspan - 1:
                                if self._has_vertical_separator(v_line_map, r, c + 1, x_coords, y_coords):
                                    is_valid_merge = False
                                    break
                        if not is_valid_merge:
                            break

                    if is_valid_merge and (rowspan > 1 or colspan > 1):
                        merged_cells.append({
                            'row': row,
                            'col': col,
                            'rowspan': rowspan,
                            'colspan': colspan,
                            'bbox': self._get_merged_bbox(row, col, rowspan, colspan, x_coords, y_coords)
                        })

                        # Mark all cells in the merge as processed
                        for r in range(row, row + rowspan):
                            for c in range(col, col + colspan):
                                processed.add((r, c))
                else:
                    processed.add((row, col))

        return merged_cells

    def _build_line_map(self, lines, axis, coords):
        """Builds a map of line segments for quick lookup."""
        line_map = {}

        if axis == 'h':
            for x1, y, x2, _ in lines:
                y_idx = self._find_nearest_coord(y, coords)
                if y_idx not in line_map:
                    line_map[y_idx] = []
                line_map[y_idx].append((min(x1, x2), max(x1, x2)))
        else:  # axis == 'v'
            for x, y1, _, y2 in lines:
                x_idx = self._find_nearest_coord(x, coords)
                if x_idx not in line_map:
                    line_map[x_idx] = []
                line_map[x_idx].append((min(y1, y2), max(y1, y2)))

        return line_map

    def _find_nearest_coord(self, value, coords, tolerance=10):
        """Finds the nearest coordinate index."""
        for i, coord in enumerate(coords):
            if abs(value - coord) <= tolerance:
                return i
        return -1

    def _has_horizontal_separator(self, h_line_map, row_idx, col_idx, x_coords, y_coords):
        """Checks if there's a horizontal line separating rows at the given column."""
        if row_idx >= len(y_coords):
            return True  # Boundary

        y_pos = y_coords[row_idx]
        y_idx = self._find_nearest_coord(y_pos, y_coords)

        if y_idx not in h_line_map:
            return False

        # Check if any line segment covers this column
        col_x1 = x_coords[col_idx]
        col_x2 = x_coords[col_idx + 1] if col_idx + 1 < len(x_coords) else col_x1 + 100

        col_center = (col_x1 + col_x2) / 2

        for line_x1, line_x2 in h_line_map[y_idx]:
            if line_x1 <= col_center <= line_x2:
                return True

        return False

    def _has_vertical_separator(self, v_line_map, row_idx, col_idx, x_coords, y_coords):
        """Checks if there's a vertical line separating columns at the given row."""
        if col_idx >= len(x_coords):
            return True  # Boundary

        x_pos = x_coords[col_idx]
        x_idx = self._find_nearest_coord(x_pos, x_coords)

        if x_idx not in v_line_map:
            return False

        # Check if any line segment covers this row
        row_y1 = y_coords[row_idx]
        row_y2 = y_coords[row_idx + 1] if row_idx + 1 < len(y_coords) else row_y1 + 100

        row_center = (row_y1 + row_y2) / 2

        for line_y1, line_y2 in v_line_map[x_idx]:
            if line_y1 <= row_center <= line_y2:
                return True

        return False

    def _get_merged_bbox(self, row, col, rowspan, colspan, x_coords, y_coords):
        """Gets the bounding box for a merged cell."""
        x1 = x_coords[col]
        y1 = y_coords[row]
        x2 = x_coords[col + colspan] if col + colspan < len(x_coords) else x_coords[-1]
        y2 = y_coords[row + rowspan] if row + rowspan < len(y_coords) else y_coords[-1]
        return [float(x1), float(y1), float(x2), float(y2)]

    def _build_cell_map(self, x_coords, y_coords, merged_cells):
        """Builds a map of all cells including merged ones."""
        # Mark merged cell origins
        merged_origins = {(m['row'], m['col']): m for m in merged_cells}

        # Mark cells that are part of a merge (not the origin)
        merged_parts = set()
        for m in merged_cells:
            for r in range(m['row'], m['row'] + m['rowspan']):
                for c in range(m['col'], m['col'] + m['colspan']):
                    if (r, c) != (m['row'], m['col']):
                        merged_parts.add((r, c))

        # Build complete cell map
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                if (row, col) in merged_parts:
                    # This cell is part of a merge, skip
                    continue

                if (row, col) in merged_origins:
                    # This is a merged cell origin
                    m = merged_origins[(row, col)]
                    self.cell_map[(row, col)] = {
                        'bbox': m['bbox'],
                        'rowspan': m['rowspan'],
                        'colspan': m['colspan'],
                        'is_merged': True
                    }
                else:
                    # Regular cell
                    x1 = x_coords[col]
                    y1 = y_coords[row]
                    x2 = x_coords[col + 1] if col + 1 < len(x_coords) else x_coords[-1]
                    y2 = y_coords[row + 1] if row + 1 < len(y_coords) else y_coords[-1]

                    self.cell_map[(row, col)] = {
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'rowspan': 1,
                        'colspan': 1,
                        'is_merged': False
                    }


def structure_to_html(cell_data_map, num_rows, num_cols):
    """
    Converts structured table data to HTML with proper rowspan/colspan.

    Args:
        cell_data_map: Dict mapping (row, col) to cell info with 'text', 'rowspan', 'colspan'
        num_rows: Number of rows
        num_cols: Number of columns

    Returns:
        HTML string representing the table
    """
    html_lines = ['<table border="1">']

    # Track which cells are covered by spans
    covered = set()

    for row in range(num_rows):
        html_lines.append('  <tr>')

        for col in range(num_cols):
            if (row, col) in covered:
                continue

            cell_info = cell_data_map.get((row, col), {})
            text = cell_info.get('text', '')
            rowspan = cell_info.get('rowspan', 1)
            colspan = cell_info.get('colspan', 1)

            # Mark covered cells
            for r in range(row, row + rowspan):
                for c in range(col, col + colspan):
                    if (r, c) != (row, col):
                        covered.add((r, c))

            # Build cell tag
            attrs = []
            if rowspan > 1:
                attrs.append(f'rowspan="{rowspan}"')
            if colspan > 1:
                attrs.append(f'colspan="{colspan}"')

            attr_str = ' ' + ' '.join(attrs) if attrs else ''
            html_lines.append(f'    <td{attr_str}>{text}</td>')

        html_lines.append('  </tr>')

    html_lines.append('</table>')

    return '\n'.join(html_lines)


def structure_to_markdown(cell_data_map, num_rows, num_cols):
    """
    Converts structured table data to Markdown.

    Note: Standard Markdown doesn't support rowspan/colspan,
    so merged cells are indicated with special markers.

    Args:
        cell_data_map: Dict mapping (row, col) to cell info
        num_rows: Number of rows
        num_cols: Number of columns

    Returns:
        Markdown string representing the table
    """
    lines = []
    covered = set()

    for row in range(num_rows):
        row_cells = []

        for col in range(num_cols):
            if (row, col) in covered:
                row_cells.append('↑')  # Indicates merged from above/left
                continue

            cell_info = cell_data_map.get((row, col), {})
            text = cell_info.get('text', '').replace('|', '\\|')
            rowspan = cell_info.get('rowspan', 1)
            colspan = cell_info.get('colspan', 1)

            # Add merge indicator if merged
            if rowspan > 1 or colspan > 1:
                text = f"{text} [↔{colspan}×↕{rowspan}]"

            # Mark covered cells
            for r in range(row, row + rowspan):
                for c in range(col, col + colspan):
                    if (r, c) != (row, col):
                        covered.add((r, c))

            row_cells.append(text)

        lines.append('| ' + ' | '.join(row_cells) + ' |')

        # Add header separator after first row
        if row == 0:
            lines.append('| ' + ' | '.join(['---'] * num_cols) + ' |')

    return '\n'.join(lines)


def extract_table_with_structure(bgr_image, binary_inv, h_lines, v_lines, ocr_func, lang='eng'):
    """
    Extracts a table with full structure information including merged cells.

    Args:
        bgr_image: Original BGR image
        binary_inv: Inverted binary image
        h_lines: Horizontal lines
        v_lines: Vertical lines
        ocr_func: OCR function to use
        lang: Language code

    Returns:
        Dict with table structure, data, and exports
    """
    analyzer = TableStructureAnalyzer()
    structure = analyzer.analyze_from_lines(h_lines, v_lines, bgr_image.shape[:2])

    if structure is None:
        return None

    # Extract text for each cell
    cell_data_map = {}

    for (row, col), cell_info in analyzer.cell_map.items():
        bbox = cell_info['bbox']
        x1, y1, x2, y2 = map(int, bbox)

        # Extract text using OCR
        if x2 > x1 and y2 > y1:
            roi = bgr_image[y1:y2, x1:x2]
            bin_roi = 255 - binary_inv[y1:y2, x1:x2]

            if roi.size > 0:
                text, conf = ocr_func(bgr_image, bin_roi, bbox, lang)
            else:
                text, conf = "", 0.0
        else:
            text, conf = "", 0.0

        cell_data_map[(row, col)] = {
            'text': text,
            'bbox': bbox,
            'rowspan': cell_info['rowspan'],
            'colspan': cell_info['colspan'],
            'is_merged': cell_info['is_merged'],
            'ocr_conf': conf
        }

    # Generate exports
    html_output = structure_to_html(cell_data_map, structure['num_rows'], structure['num_cols'])
    markdown_output = structure_to_markdown(cell_data_map, structure['num_rows'], structure['num_cols'])

    # Also create DataFrame (flattened for merged cells)
    df = _create_dataframe_with_merges(cell_data_map, structure['num_rows'], structure['num_cols'])

    return {
        'num_rows': structure['num_rows'],
        'num_cols': structure['num_cols'],
        'cell_data': cell_data_map,
        'merged_cells': structure['merged_cells'],
        'html': html_output,
        'markdown': markdown_output,
        'dataframe': df
    }


def _create_dataframe_with_merges(cell_data_map, num_rows, num_cols):
    """Creates a DataFrame handling merged cells."""
    data = [["" for _ in range(num_cols)] for _ in range(num_rows)]
    covered = set()

    for (row, col), cell_info in cell_data_map.items():
        text = cell_info.get('text', '')
        rowspan = cell_info.get('rowspan', 1)
        colspan = cell_info.get('colspan', 1)

        # Fill the primary cell
        if 0 <= row < num_rows and 0 <= col < num_cols:
            data[row][col] = text

        # Fill merged cells with reference marker
        for r in range(row, row + rowspan):
            for c in range(col, col + colspan):
                if (r, c) != (row, col) and 0 <= r < num_rows and 0 <= c < num_cols:
                    data[r][c] = f"[→{row},{col}]"  # Reference to origin cell
                    covered.add((r, c))

    return pd.DataFrame(data)
