import cv2
import numpy as np

def clean_stone_inscription(image_path: str) -> np.ndarray:
    """
    Decouples stone background textures from etched historical glyphs.
    """
    # 1. Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")
        
    # 2. Anisotropic Diffusion / Bilateral Filter: Kill rock grain, keep edges
    # d=9 (pixel neighborhood), sigmaColor=75 (large values mix colors), sigmaSpace=75
    smoothed = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    
    # 3. CLAHE: Maximize the micro-shadow gradients inside the shallow carvings
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(smoothed)
    
    # 4. Adaptive Thresholding: Local window processing prevents lightning issues
    # Uses a local 11x11 window to handle micro-surface deviations
    binarized = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    return binarized
