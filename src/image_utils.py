# src/image_utils.py

import cv2
import numpy as np
from PIL import Image

def load_image_as_bgr(path_or_object):
    """Loads an image file or PIL Image object as an OpenCV matrix in BGR format."""
    try:
        if isinstance(path_or_object, str):
            img = Image.open(path_or_object).convert("RGB")
        elif isinstance(path_or_object, Image.Image):
            img = path_or_object.convert("RGB")
        else:
            raise TypeError("Input must be a file path (str) or a PIL Image object.")
        
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def remove_lines(bgr_image, h_lines, v_lines, line_thickness=3):
    """
    Removes (paints white) the given horizontal and vertical lines from the image.
    """
    img_no_lines = bgr_image.copy()
    
    if h_lines:
        for x1, y1, x2, y2 in h_lines:
            cv2.line(img_no_lines, (x1, y1), (x2, y2), (255, 255, 255), line_thickness)
        
    if v_lines:
        for x1, y1, x2, y2 in v_lines:
            cv2.line(img_no_lines, (x1, y1), (x2, y2), (255, 255, 255), line_thickness)
        
    return img_no_lines

def deskew(gray_image):
    """Corrects small skews in the image using Hough Transform."""
    edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    if lines is None:
        return gray_image

    angles = []
    for line in lines:
        rho, theta = line[0]
        angle = np.rad2deg(theta) - 90
        if -45 < angle < 45:
            angles.append(angle)

    if not angles:
        return gray_image

    median_angle = np.median(angles)
    h, w = gray_image.shape
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    deskewed = cv2.warpAffine(gray_image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    return deskewed


def analyze_image_quality(bgr_image):
    """
    Analyzes image quality to determine optimal preprocessing parameters.

    Returns a dictionary with:
    - laplacian_variance: Measure of image sharpness/noise (higher = sharper/noisier)
    - brightness: Average brightness (0-255)
    - contrast: Standard deviation of pixel values
    - quality_preset: Recommended preprocessing preset ('clean', 'normal', 'noisy', 'blurry')
    - denoising_h: Recommended denoising strength
    - clahe_clip_limit: Recommended CLAHE clip limit
    """
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

    # Calculate Laplacian variance (sharpness/noise indicator)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_var = laplacian.var()

    # Calculate brightness and contrast
    brightness = np.mean(gray)
    contrast = np.std(gray)

    # Determine quality preset based on metrics
    quality_preset = 'normal'
    denoising_h = 7  # Default
    clahe_clip_limit = 2.0  # Default

    if laplacian_var > 1500:
        # High variance: Very sharp or very noisy image
        quality_preset = 'noisy'
        denoising_h = 12
        clahe_clip_limit = 1.5
    elif laplacian_var > 800:
        # Medium-high variance: Sharp image with some noise
        quality_preset = 'normal'
        denoising_h = 7
        clahe_clip_limit = 2.0
    elif laplacian_var > 200:
        # Medium variance: Clean image
        quality_preset = 'clean'
        denoising_h = 3
        clahe_clip_limit = 2.5
    else:
        # Low variance: Blurry image
        quality_preset = 'blurry'
        denoising_h = 5  # Light denoising to avoid further blur
        clahe_clip_limit = 3.0  # Higher contrast enhancement

    # Adjust for brightness
    if brightness < 100:
        # Dark image - increase contrast enhancement
        clahe_clip_limit = min(clahe_clip_limit + 1.0, 4.0)
    elif brightness > 200:
        # Very bright image - reduce contrast enhancement
        clahe_clip_limit = max(clahe_clip_limit - 0.5, 1.0)

    return {
        'laplacian_variance': float(laplacian_var),
        'brightness': float(brightness),
        'contrast': float(contrast),
        'quality_preset': quality_preset,
        'denoising_h': denoising_h,
        'clahe_clip_limit': clahe_clip_limit
    }


def preprocess_for_ocr(bgr_image, denoising_h=7, clahe_clip_limit=2.0,
                       adaptive_block_size=35, adaptive_c=11):
    """
    Applies preprocessing pipeline for OCR with configurable parameters.

    Args:
        bgr_image: Input BGR image
        denoising_h: Denoising strength (higher = more denoising)
        clahe_clip_limit: CLAHE clip limit for contrast enhancement
        adaptive_block_size: Block size for adaptive thresholding
        adaptive_c: Constant for adaptive thresholding
    """
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

    # Apply denoising with configurable strength
    if denoising_h > 0:
        gray = cv2.fastNlMeansDenoising(gray, h=denoising_h)

    gray = deskew(gray)

    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    bin_img = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        adaptive_block_size, adaptive_c
    )

    return bin_img

def find_document_corners(bgr_image):
    """Finds the corners of the largest rectangle in the image (for perspective correction)."""
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 75, 200)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

    if len(approx) == 4:
        return approx.reshape(4, 2).astype("float32")
    return None

def four_point_transform(image, pts):
    """Applies perspective correction to the image based on the found four corners."""
    rect = np.array(pts, dtype="float32")
    (tl, tr, br, bl) = rect
    
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = int(max(width_a, width_b))
    
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = int(max(height_a, height_b))
    
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")
        
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))
    return warped