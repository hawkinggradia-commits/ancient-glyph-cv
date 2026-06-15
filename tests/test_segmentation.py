import numpy as np
from src.segmentation import segment_glyphs

def test_segment_glyphs_returns_list():
    # Create a blank black image
    dummy_bin = np.zeros((200, 200), dtype=np.uint8)
    
    # Draw a white square that should be detected as a glyph
    # Area = 50x50 = 2500 (falls between min_area=50 and max_area=5000)
    # Aspect Ratio = 1.0
    dummy_bin[50:100, 50:100] = 255
    
    # Draw a tiny noise spec that should be ignored
    # Area = 2x2 = 4 (fails min_area)
    dummy_bin[10:12, 10:12] = 255
    
    # Draw a long crack that should be ignored
    # Area = 100x5 = 500
    # Aspect Ratio = 5/100 = 0.05 (fails aspect ratio rule < 0.2)
    dummy_bin[50:150, 10:15] = 255

    glyphs = segment_glyphs(dummy_bin, min_area=50, max_area=5000)
    
    assert isinstance(glyphs, list)
    # Only the 50x50 square should pass the heuristics
    assert len(glyphs) == 1
    assert glyphs[0].shape == (50, 50)
