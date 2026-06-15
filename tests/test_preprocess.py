import os
import cv2
import numpy as np
import pytest
from src.preprocess import clean_stone_inscription

def test_clean_stone_inscription_returns_binary_array():
    # Create a dummy image for testing
    test_img_path = "test_dummy.jpg"
    dummy_img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    cv2.imwrite(test_img_path, dummy_img)
    
    try:
        result = clean_stone_inscription(test_img_path)
        
        # Verify it returns a numpy array
        assert isinstance(result, np.ndarray)
        # Verify shape is maintained
        assert result.shape == (100, 100)
        # Verify it is a binary image (values should be 0 or 255 typically, but testing dtype is safer)
        assert result.dtype == np.uint8
    finally:
        if os.path.exists(test_img_path):
            os.remove(test_img_path)

def test_clean_stone_inscription_handles_missing_file():
    with pytest.raises(FileNotFoundError):
        clean_stone_inscription("non_existent_file_xyz.jpg")
