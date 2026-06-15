import cv2
import numpy as np

def segment_glyphs(binarized_img: np.ndarray, min_area=50, max_area=5000) -> list:
    """
    Finds bounding boxes around connected components and filters out anomalies.
    """
    # Find all distinct bounding shapes
    contours, _ = cv2.findContours(binarized_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_glyph_crops = []
    
    # Optional: Apply morphological closing to bridge fractures before contouring
    kernel_sq = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed_img = cv2.morphologyEx(binarized_img, cv2.MORPH_CLOSE, kernel_sq, iterations=1)
    
    contours, _ = cv2.findContours(closed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        aspect_ratio = float(w) / h
        
        # Heuristic rules: Exclude massive cracks or microscopic rock flecks
        if min_area < area < max_area:
            if 0.2 < aspect_ratio < 5.0:  # Prevents long horizontal/vertical cracks
                # Crop from the closed binary image to ensure contiguous glyph shapes
                crop = closed_img[y:y+h, x:x+w]
                valid_glyph_crops.append(crop)
                
    return valid_glyph_crops
