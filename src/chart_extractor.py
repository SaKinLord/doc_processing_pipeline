# src/chart_extractor.py

"""
Chart Data Extraction Module

Extracts numerical data from visual charts using image analysis:
- Bar charts: Extract bar heights/widths as values
- Line charts: Extract data points along lines
- Pie charts: Extract segment angles/percentages
"""

import cv2
import numpy as np
from collections import defaultdict


class ChartDataExtractor:
    """
    Extracts numerical data from chart images using computer vision.
    """

    def __init__(self):
        self.supported_types = ['bar_chart', 'line_chart', 'pie_chart', 'scatter_plot']

    def extract_data(self, roi_bgr, chart_type):
        """
        Main entry point for chart data extraction.

        Args:
            roi_bgr: BGR image of the chart region
            chart_type: Type of chart ('bar_chart', 'line_chart', etc.)

        Returns:
            dict with extracted data and metadata
        """
        if roi_bgr is None or roi_bgr.size == 0:
            return {'success': False, 'error': 'Empty image'}

        if chart_type not in self.supported_types:
            return {'success': False, 'error': f'Unsupported chart type: {chart_type}'}

        try:
            if chart_type == 'bar_chart':
                return self._extract_bar_chart(roi_bgr)
            elif chart_type == 'line_chart':
                return self._extract_line_chart(roi_bgr)
            elif chart_type == 'pie_chart':
                return self._extract_pie_chart(roi_bgr)
            elif chart_type == 'scatter_plot':
                return self._extract_scatter_plot(roi_bgr)
            else:
                return {'success': False, 'error': f'Not implemented: {chart_type}'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _extract_bar_chart(self, roi_bgr):
        """
        Extracts data from bar charts.

        Strategy:
        1. Detect vertical/horizontal bars using contour analysis
        2. Measure bar heights/widths
        3. Estimate values based on chart bounds
        """
        h, w = roi_bgr.shape[:2]
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

        # Detect chart orientation (vertical or horizontal bars)
        orientation = self._detect_bar_orientation(roi_bgr)

        # Find bars using color segmentation
        bars = self._find_bars(roi_bgr, orientation)

        if not bars:
            return {
                'success': False,
                'error': 'No bars detected',
                'chart_type': 'bar_chart'
            }

        # Estimate axis bounds
        axis_info = self._estimate_axis_bounds(roi_bgr, bars, orientation)

        # Calculate relative values
        values = []
        for bar in bars:
            if orientation == 'vertical':
                # Bar height relative to chart height
                bar_height = bar['bbox'][3] - bar['bbox'][1]
                relative_value = bar_height / h
            else:
                # Bar width relative to chart width
                bar_width = bar['bbox'][2] - bar['bbox'][0]
                relative_value = bar_width / w

            values.append({
                'index': bar['index'],
                'bbox': bar['bbox'],
                'relative_value': round(relative_value, 4),
                'color': bar.get('color', 'unknown')
            })

        # Sort by position
        if orientation == 'vertical':
            values.sort(key=lambda x: x['bbox'][0])  # Sort by x position
        else:
            values.sort(key=lambda x: x['bbox'][1])  # Sort by y position

        return {
            'success': True,
            'chart_type': 'bar_chart',
            'orientation': orientation,
            'num_bars': len(values),
            'values': values,
            'axis_info': axis_info,
            'chart_bounds': {'width': w, 'height': h}
        }

    def _extract_line_chart(self, roi_bgr):
        """
        Extracts data points from line charts.

        Strategy:
        1. Detect line using edge detection and Hough transform
        2. Sample points along the line
        3. Convert y-coordinates to values
        """
        h, w = roi_bgr.shape[:2]
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

        # Edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Find lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                                 minLineLength=30, maxLineGap=10)

        if lines is None or len(lines) == 0:
            return {
                'success': False,
                'error': 'No lines detected',
                'chart_type': 'line_chart'
            }

        # Filter for non-axis lines (not perfectly horizontal/vertical)
        data_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

            # Skip axis lines (0° or 90°)
            if 5 < angle < 85:
                data_lines.append(line[0])

        # Sample points from detected lines
        data_points = []
        for i, (x1, y1, x2, y2) in enumerate(data_lines[:5]):  # Limit to 5 lines
            # Sample 10 points along each line
            for t in np.linspace(0, 1, 10):
                x = int(x1 + t * (x2 - x1))
                y = int(y1 + t * (y2 - y1))
                # Convert y to value (top = high, bottom = low)
                relative_value = 1.0 - (y / h)
                data_points.append({
                    'x': x,
                    'y': y,
                    'relative_x': round(x / w, 4),
                    'relative_value': round(relative_value, 4),
                    'line_index': i
                })

        # Sort by x position
        data_points.sort(key=lambda p: p['x'])

        return {
            'success': True,
            'chart_type': 'line_chart',
            'num_lines': len(data_lines),
            'num_points': len(data_points),
            'data_points': data_points,
            'chart_bounds': {'width': w, 'height': h}
        }

    def _extract_pie_chart(self, roi_bgr):
        """
        Extracts segment data from pie charts.

        Strategy:
        1. Detect circles/ellipses
        2. Segment by color
        3. Calculate angles for each segment
        """
        h, w = roi_bgr.shape[:2]

        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

        # Find dominant colors (segments)
        segments = self._segment_by_color(roi_bgr)

        if len(segments) < 2:
            return {
                'success': False,
                'error': 'Could not segment pie chart',
                'chart_type': 'pie_chart'
            }

        # Calculate percentage for each segment
        total_pixels = sum(seg['pixel_count'] for seg in segments)

        for seg in segments:
            seg['percentage'] = round(seg['pixel_count'] / total_pixels * 100, 2)
            seg['angle'] = round(seg['percentage'] / 100 * 360, 2)

        # Sort by percentage (largest first)
        segments.sort(key=lambda x: x['percentage'], reverse=True)

        return {
            'success': True,
            'chart_type': 'pie_chart',
            'num_segments': len(segments),
            'segments': segments,
            'chart_bounds': {'width': w, 'height': h}
        }

    def _extract_scatter_plot(self, roi_bgr):
        """
        Extracts data points from scatter plots.

        Strategy:
        1. Detect circular markers using blob detection
        2. Record positions as data points
        """
        h, w = roi_bgr.shape[:2]
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

        # Setup blob detector
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 10
        params.maxArea = 500
        params.filterByCircularity = True
        params.minCircularity = 0.5

        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(255 - gray)  # Invert for dark markers

        if not keypoints:
            # Try with original image
            keypoints = detector.detect(gray)

        data_points = []
        for i, kp in enumerate(keypoints):
            x, y = kp.pt
            data_points.append({
                'index': i,
                'x': round(x, 2),
                'y': round(y, 2),
                'relative_x': round(x / w, 4),
                'relative_y': round(1.0 - y / h, 4),  # Invert y-axis
                'size': round(kp.size, 2)
            })

        # Sort by x position
        data_points.sort(key=lambda p: p['x'])

        return {
            'success': True,
            'chart_type': 'scatter_plot',
            'num_points': len(data_points),
            'data_points': data_points,
            'chart_bounds': {'width': w, 'height': h}
        }

    def _detect_bar_orientation(self, roi_bgr):
        """Detects if bars are vertical or horizontal."""
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Count horizontal vs vertical edges
        h_kernel = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        v_kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])

        h_edges = cv2.filter2D(edges, -1, h_kernel)
        v_edges = cv2.filter2D(edges, -1, v_kernel)

        if np.sum(h_edges) > np.sum(v_edges):
            return 'horizontal'
        else:
            return 'vertical'

    def _find_bars(self, roi_bgr, orientation):
        """Finds individual bars in the chart."""
        h, w = roi_bgr.shape[:2]
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

        # Threshold to find filled regions
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bars = []
        for i, cnt in enumerate(contours):
            x, y, bw, bh = cv2.boundingRect(cnt)
            area = bw * bh

            # Filter by shape (bars are rectangular)
            aspect_ratio = bw / bh if bh > 0 else 0

            if orientation == 'vertical':
                # Vertical bars: taller than wide
                if aspect_ratio < 2.0 and bh > 20 and area > 100:
                    # Get dominant color of this bar
                    mask = np.zeros(gray.shape, dtype=np.uint8)
                    cv2.drawContours(mask, [cnt], -1, 255, -1)
                    color = self._get_dominant_color(roi_bgr, mask)

                    bars.append({
                        'index': i,
                        'bbox': [x, y, x + bw, y + bh],
                        'width': bw,
                        'height': bh,
                        'color': color
                    })
            else:
                # Horizontal bars: wider than tall
                if aspect_ratio > 0.5 and bw > 20 and area > 100:
                    mask = np.zeros(gray.shape, dtype=np.uint8)
                    cv2.drawContours(mask, [cnt], -1, 255, -1)
                    color = self._get_dominant_color(roi_bgr, mask)

                    bars.append({
                        'index': i,
                        'bbox': [x, y, x + bw, y + bh],
                        'width': bw,
                        'height': bh,
                        'color': color
                    })

        return bars

    def _estimate_axis_bounds(self, roi_bgr, bars, orientation):
        """Estimates the axis value range based on bar positions."""
        if not bars:
            return {}

        if orientation == 'vertical':
            # Y-axis bounds
            min_y = min(bar['bbox'][1] for bar in bars)
            max_y = max(bar['bbox'][3] for bar in bars)
            return {
                'y_min_pixel': min_y,
                'y_max_pixel': max_y,
                'estimated_range': max_y - min_y
            }
        else:
            # X-axis bounds
            min_x = min(bar['bbox'][0] for bar in bars)
            max_x = max(bar['bbox'][2] for bar in bars)
            return {
                'x_min_pixel': min_x,
                'x_max_pixel': max_x,
                'estimated_range': max_x - min_x
            }

    def _segment_by_color(self, roi_bgr):
        """Segments image by dominant colors (for pie charts)."""
        # Quantize colors
        pixels = roi_bgr.reshape(-1, 3).astype(np.float32)

        # K-means clustering
        k = 6  # Max number of segments
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

        try:
            _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10,
                                             cv2.KMEANS_RANDOM_CENTERS)
        except:
            return []

        # Count pixels per cluster
        segments = []
        for i in range(k):
            count = np.sum(labels == i)
            if count > 100:  # Minimum segment size
                color = centers[i].astype(int).tolist()
                segments.append({
                    'index': i,
                    'color_bgr': color,
                    'color_name': self._color_to_name(color),
                    'pixel_count': int(count)
                })

        return segments

    def _get_dominant_color(self, roi_bgr, mask):
        """Gets the dominant color in the masked region."""
        masked = cv2.bitwise_and(roi_bgr, roi_bgr, mask=mask)
        pixels = masked[mask > 0]

        if len(pixels) == 0:
            return 'unknown'

        mean_color = np.mean(pixels, axis=0).astype(int)
        return self._color_to_name(mean_color.tolist())

    def _color_to_name(self, bgr):
        """Converts BGR color to a simple name."""
        b, g, r = bgr

        # Simple color classification
        if r > 200 and g < 100 and b < 100:
            return 'red'
        elif r < 100 and g > 200 and b < 100:
            return 'green'
        elif r < 100 and g < 100 and b > 200:
            return 'blue'
        elif r > 200 and g > 200 and b < 100:
            return 'yellow'
        elif r > 200 and g < 100 and b > 200:
            return 'magenta'
        elif r < 100 and g > 200 and b > 200:
            return 'cyan'
        elif r > 200 and g > 150 and b < 100:
            return 'orange'
        elif r > 150 and g > 150 and b > 150:
            return 'gray'
        elif r < 50 and g < 50 and b < 50:
            return 'black'
        elif r > 200 and g > 200 and b > 200:
            return 'white'
        else:
            return f'rgb({r},{g},{b})'


# Convenience function
def extract_chart_data(roi_bgr, chart_type):
    """
    Extracts numerical data from a chart image.

    Args:
        roi_bgr: BGR image of the chart
        chart_type: Type of chart

    Returns:
        dict with extracted data
    """
    extractor = ChartDataExtractor()
    return extractor.extract_data(roi_bgr, chart_type)
