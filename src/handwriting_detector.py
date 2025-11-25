# src/handwriting_detector.py

"""
Handwriting Detection Module

Classifies text regions as 'printed' or 'handwritten' to route OCR appropriately.
Uses a combination of:
1. Texture analysis (Gabor filters, LBP patterns)
2. Morphological features (stroke variance, connectivity)
3. Statistical features (intensity distribution, edge patterns)
"""

import cv2
import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks


class HandwritingDetector:
    """
    Lightweight classifier to determine if text is handwritten or printed.

    Uses texture and morphological features to make the determination.
    This is much faster than running multiple OCR engines.
    """

    def __init__(self, confidence_threshold=0.55):
        """
        Initialize the handwriting detector.

        Args:
            confidence_threshold: Threshold for classification confidence (0-1)
        """
        self.confidence_threshold = confidence_threshold
        self._model = None
        self._use_ml_model = False

        # Try to load a pre-trained model if available
        self._try_load_model()

    def _try_load_model(self):
        """Attempts to load a pre-trained ML model for handwriting detection."""
        try:
            # Try to use a lightweight CNN from torchvision
            import torch
            import torchvision.models as models

            # Check if we have a custom trained model
            # For now, we'll use feature-based classification
            self._use_ml_model = False
            print("    - Handwriting detector initialized (feature-based mode)")
        except ImportError:
            self._use_ml_model = False
            print("    - Handwriting detector initialized (feature-based mode)")

    def _extract_texture_features(self, gray_image):
        """
        Extracts texture features from the image.

        Handwritten text typically has:
        - Higher variance in stroke width
        - More irregular edge patterns
        - Less uniform spacing
        """
        features = {}

        if gray_image.size == 0:
            return features

        # 1. Laplacian variance (edge sharpness)
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
        features['laplacian_var'] = np.var(laplacian)
        features['laplacian_mean'] = np.mean(np.abs(laplacian))

        # 2. Gradient analysis
        grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        features['gradient_mean'] = np.mean(gradient_magnitude)
        features['gradient_std'] = np.std(gradient_magnitude)

        # 3. Local Binary Pattern (simplified)
        # Handwritten text has more varied local patterns
        lbp_sum = 0
        h, w = gray_image.shape
        if h > 2 and w > 2:
            for i in range(1, min(h-1, 50)):
                for j in range(1, min(w-1, 50)):
                    center = gray_image[i, j]
                    pattern = 0
                    if gray_image[i-1, j] > center: pattern += 1
                    if gray_image[i+1, j] > center: pattern += 2
                    if gray_image[i, j-1] > center: pattern += 4
                    if gray_image[i, j+1] > center: pattern += 8
                    lbp_sum += pattern
            features['lbp_complexity'] = lbp_sum / (min(h-2, 49) * min(w-2, 49))
        else:
            features['lbp_complexity'] = 0

        return features

    def _extract_morphological_features(self, binary_image):
        """
        Extracts morphological features from binarized text.

        Handwritten text typically has:
        - Variable stroke width
        - Connected components of varying sizes
        - Less regular baselines
        """
        features = {}

        if binary_image.size == 0:
            return features

        # Ensure binary
        _, binary = cv2.threshold(binary_image, 127, 255, cv2.THRESH_BINARY)

        # 1. Connected component analysis
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            255 - binary, connectivity=8
        )

        if num_labels > 1:
            # Exclude background (label 0)
            areas = stats[1:, cv2.CC_STAT_AREA]
            widths = stats[1:, cv2.CC_STAT_WIDTH]
            heights = stats[1:, cv2.CC_STAT_HEIGHT]

            features['num_components'] = num_labels - 1
            features['area_variance'] = np.var(areas) if len(areas) > 1 else 0
            features['size_variance'] = np.var(widths * heights) if len(areas) > 1 else 0

            # Aspect ratio variance (handwriting has more varied shapes)
            aspect_ratios = widths / (heights + 1e-5)
            features['aspect_ratio_var'] = np.var(aspect_ratios) if len(areas) > 1 else 0

            # Average component size
            features['avg_component_area'] = np.mean(areas)
        else:
            features['num_components'] = 0
            features['area_variance'] = 0
            features['size_variance'] = 0
            features['aspect_ratio_var'] = 0
            features['avg_component_area'] = 0

        # 2. Stroke width estimation
        # Use distance transform to estimate stroke width
        dist_transform = cv2.distanceTransform(255 - binary, cv2.DIST_L2, 5)
        stroke_widths = dist_transform[dist_transform > 0]

        if len(stroke_widths) > 0:
            features['stroke_width_mean'] = np.mean(stroke_widths)
            features['stroke_width_std'] = np.std(stroke_widths)
            features['stroke_width_var_ratio'] = (
                features['stroke_width_std'] / (features['stroke_width_mean'] + 1e-5)
            )
        else:
            features['stroke_width_mean'] = 0
            features['stroke_width_std'] = 0
            features['stroke_width_var_ratio'] = 0

        # 3. Horizontal projection profile variance
        # Handwriting has more irregular baselines
        h_projection = np.sum(255 - binary, axis=1)
        features['h_projection_var'] = np.var(h_projection)

        # 4. Vertical projection profile variance
        v_projection = np.sum(255 - binary, axis=0)
        features['v_projection_var'] = np.var(v_projection)

        return features

    def _extract_stroke_features(self, gray_image):
        """
        Extract stroke-specific features that better distinguish handwriting.

        Handwritten text has:
        - Variable stroke width (irregular pen pressure)
        - Irregular baseline (letters don't align perfectly)
        - Variable character spacing and connectivity

        Args:
            gray_image: Grayscale image

        Returns:
            dict: Stroke feature metrics
        """
        features = {}

        if gray_image.size == 0:
            return features

        # 1. Stroke width variation using distance transform
        _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

        if dist_transform.size > 0 and np.any(dist_transform > 0):
            stroke_widths = dist_transform[dist_transform > 0]
            features['stroke_width_mean'] = np.mean(stroke_widths)
            features['stroke_width_std'] = np.std(stroke_widths)
            # High std = variable stroke width = likely handwriting
            features['stroke_variation_ratio'] = features['stroke_width_std'] / (features['stroke_width_mean'] + 1e-6)
        else:
            features['stroke_width_mean'] = 0
            features['stroke_width_std'] = 0
            features['stroke_variation_ratio'] = 0

        # 2. Baseline irregularity
        # Find text baseline by horizontal projection
        h_proj = np.sum(binary, axis=1)
        if len(h_proj) > 10:
            # Find peaks in projection (text lines)
            try:
                peaks, _ = find_peaks(h_proj, height=np.max(h_proj) * 0.3)
                if len(peaks) > 1:
                    peak_diffs = np.diff(peaks)
                    features['baseline_regularity'] = np.std(peak_diffs) / (np.mean(peak_diffs) + 1e-6)
                else:
                    features['baseline_regularity'] = 0
            except:
                features['baseline_regularity'] = 0
        else:
            features['baseline_regularity'] = 0

        # 3. Character connectivity - component analysis
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]  # Exclude background
            features['component_count'] = num_labels - 1
            features['avg_component_area'] = np.mean(areas)
            features['component_area_variance'] = np.std(areas) / (np.mean(areas) + 1e-6)
        else:
            features['component_count'] = 0
            features['avg_component_area'] = 0
            features['component_area_variance'] = 0

        return features

    def _extract_statistical_features(self, gray_image):
        """
        Extracts statistical features from pixel intensity distribution.
        """
        features = {}

        if gray_image.size == 0:
            return features

        # Intensity statistics
        features['intensity_mean'] = np.mean(gray_image)
        features['intensity_std'] = np.std(gray_image)
        features['intensity_skew'] = self._calculate_skewness(gray_image.flatten())
        features['intensity_kurtosis'] = self._calculate_kurtosis(gray_image.flatten())

        # Histogram features
        hist = cv2.calcHist([gray_image], [0], None, [32], [0, 256])
        hist = hist.flatten() / hist.sum()

        # Entropy
        hist_nonzero = hist[hist > 0]
        features['histogram_entropy'] = -np.sum(hist_nonzero * np.log2(hist_nonzero))

        # Peak count (printed text has fewer, sharper peaks)
        features['histogram_peaks'] = np.sum((hist[1:-1] > hist[:-2]) & (hist[1:-1] > hist[2:]))

        return features

    def _calculate_skewness(self, data):
        """Calculate skewness of data."""
        n = len(data)
        if n < 3:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)

    def _calculate_kurtosis(self, data):
        """Calculate kurtosis of data."""
        n = len(data)
        if n < 4:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3

    def _score_handwriting_likelihood(self, features):
        """
        Calculates a handwriting likelihood score based on extracted features.

        Returns:
            score: 0.0 (definitely printed) to 1.0 (definitely handwritten)
        """
        score = 0.0
        weights_sum = 0.0

        # 1. Stroke variation ratio (NEW - high variance = handwritten)
        if 'stroke_variation_ratio' in features:
            svr = features['stroke_variation_ratio']
            if svr > 0.4:  # High variation = handwriting
                score += 0.3 * min(svr / 0.8, 1.0)
            elif svr < 0.2:  # Low variation = printed
                score -= 0.2
            weights_sum += 0.3
        # Fallback to old feature name if new one not present
        elif 'stroke_width_var_ratio' in features:
            var_ratio = features['stroke_width_var_ratio']
            if var_ratio > 0.5:
                score += 0.3 * min(var_ratio / 0.8, 1.0)
            else:
                score -= 0.15
            weights_sum += 0.3

        # 1b. Baseline irregularity (NEW - irregular baseline = handwriting)
        if 'baseline_regularity' in features:
            br = features['baseline_regularity']
            if br > 0.3:  # Irregular baseline = handwriting
                score += 0.25 * min(br / 1.0, 1.0)
            elif br < 0.1:  # Regular baseline = printed
                score -= 0.15
            weights_sum += 0.25

        # 2. Component area variance (NEW - high variance = handwritten)
        if 'component_area_variance' in features:
            cav = features['component_area_variance']
            if cav > 0.8:  # Highly variable component sizes = handwriting
                score += 0.25 * min(cav / 2.0, 1.0)
            elif cav < 0.3:  # Uniform components = printed
                score -= 0.15
            weights_sum += 0.25
        # Fallback to old feature name
        elif 'area_variance' in features and features.get('num_components', 0) > 3:
            # Normalize by average area
            if features.get('avg_component_area', 0) > 0:
                normalized_var = features['area_variance'] / (features['avg_component_area'] ** 2)
                if normalized_var > 0.5:
                    score += 0.2 * min(normalized_var / 2.0, 1.0)
            weights_sum += 0.2

        # 3. Aspect ratio variance (handwriting has varied character shapes)
        if 'aspect_ratio_var' in features:
            if features['aspect_ratio_var'] > 0.3:
                score += 0.15 * min(features['aspect_ratio_var'] / 1.0, 1.0)
            weights_sum += 0.15

        # 4. Horizontal projection variance (irregular baselines = handwritten)
        if 'h_projection_var' in features:
            # High variance indicates irregular baseline (handwriting)
            h_var = features['h_projection_var']
            if h_var > 10000:
                score += 0.15 * min(h_var / 50000, 1.0)
            weights_sum += 0.15

        # 5. LBP complexity (more varied patterns = handwritten)
        if 'lbp_complexity' in features:
            complexity = features['lbp_complexity']
            if complexity > 6:
                score += 0.1 * min((complexity - 6) / 4, 1.0)
            weights_sum += 0.1

        # 6. Gradient variance (inconsistent strokes = handwritten)
        if 'gradient_std' in features and 'gradient_mean' in features:
            if features['gradient_mean'] > 0:
                grad_cv = features['gradient_std'] / features['gradient_mean']
                if grad_cv > 0.8:
                    score += 0.1 * min(grad_cv / 1.5, 1.0)
            weights_sum += 0.1

        # 7. Histogram entropy (higher entropy = more irregular = handwritten)
        if 'histogram_entropy' in features:
            entropy = features['histogram_entropy']
            if entropy > 3.5:
                score += 0.05 * min((entropy - 3.5) / 2, 1.0)
            weights_sum += 0.05

        # NEW: 8. Cursive-ness check (connected components width vs height)
        # Handwriting often has wider connected components (connected letters)
        if 'avg_component_area' in features and features.get('num_components', 0) > 2:
            # We don't have direct width/height stats here, but we can infer from aspect ratio
            # If we had access to raw stats we could check width > height * 1.5
            pass

        # Normalize score
        if weights_sum > 0:
            score = score / weights_sum

        return min(max(score, 0.0), 1.0)

    def classify(self, roi_image):
        """
        Classifies a text region as printed or handwritten.

        Args:
            roi_image: Input image (BGR or grayscale)

        Returns:
            dict with:
                - 'classification': 'printed' or 'handwritten'
                - 'confidence': confidence score (0-1)
                - 'features': extracted features (for debugging)
        """
        if roi_image is None or roi_image.size == 0:
            return {
                'classification': 'printed',
                'confidence': 0.5,
                'features': {}
            }

        # Convert to grayscale if needed
        if len(roi_image.shape) == 3:
            gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi_image.copy()

        # Skip very small images
        if gray.shape[0] < 10 or gray.shape[1] < 10:
            return {
                'classification': 'printed',
                'confidence': 0.5,
                'features': {}
            }

        # Extract features
        texture_features = self._extract_texture_features(gray)

        # Binarize for morphological analysis
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        morph_features = self._extract_morphological_features(binary)

        stat_features = self._extract_statistical_features(gray)

        # NEW: Extract stroke-specific features for better handwriting detection
        stroke_features = self._extract_stroke_features(gray)

        # Combine all features
        all_features = {**texture_features, **morph_features, **stat_features, **stroke_features}

        # Calculate handwriting score
        handwriting_score = self._score_handwriting_likelihood(all_features)

        # Determine classification
        if handwriting_score > self.confidence_threshold:
            classification = 'handwritten'
            confidence = handwriting_score
        else:
            classification = 'printed'
            confidence = 1.0 - handwriting_score

        return {
            'classification': classification,
            'confidence': confidence,
            'features': all_features,
            'handwriting_score': handwriting_score
        }

    def is_handwritten(self, roi_image):
        """
        Simple boolean check for handwriting.

        Args:
            roi_image: Input image

        Returns:
            True if handwritten, False if printed
        """
        result = self.classify(roi_image)
        return result['classification'] == 'handwritten'


# Global instance (lazy initialization)
_detector_instance = None


def get_handwriting_detector():
    """Gets or creates the global handwriting detector instance."""
    global _detector_instance

    if _detector_instance is None:
        # Try to get from ModelManager first
        try:
            from .model_manager import ModelManager
            detector = ModelManager.get_handwriting_detector()
            if detector is not None:
                _detector_instance = detector
                return _detector_instance
        except Exception:
            pass

        # Fall back to creating new instance
        _detector_instance = HandwritingDetector()

    return _detector_instance


def classify_text_type(roi_image):
    """
    Convenience function to classify text type.

    Args:
        roi_image: Input image (BGR or grayscale)

    Returns:
        'printed' or 'handwritten'
    """
    detector = get_handwriting_detector()
    result = detector.classify(roi_image)
    return result['classification']


def get_text_type_confidence(roi_image):
    """
    Get classification with confidence score.

    Args:
        roi_image: Input image

    Returns:
        tuple: (classification, confidence)
    """
    detector = get_handwriting_detector()
    result = detector.classify(roi_image)
    return result['classification'], result['confidence']