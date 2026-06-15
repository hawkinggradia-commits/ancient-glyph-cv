import numpy as np
import pytest
from src.clustering import FeatureExtractor, evaluate_and_cluster

def test_feature_extractor_returns_correct_dimensions():
    extractor = FeatureExtractor(batch_size=2)
    # Create two dummy images representing glyph crops
    dummy_glyph_1 = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    dummy_glyph_2 = np.random.randint(0, 256, (40, 50), dtype=np.uint8)
    
    features = extractor.extract([dummy_glyph_1, dummy_glyph_2])
    
    # 2 images passed in, so we expect 2 rows
    assert features.shape[0] == 2
    # ResNet18 spatial embeddings should be 512D
    assert features.shape[1] == 512

def test_evaluate_and_cluster():
    # Generate 50 points of dummy 512D features
    np.random.seed(42)
    dummy_features = np.random.rand(50, 512)
    
    # Cluster
    labels, optimal_eps = evaluate_and_cluster(dummy_features, min_samples=2)
    
    assert isinstance(labels, np.ndarray)
    assert len(labels) == 50
    # Epsilon should be a float from the search range
    assert optimal_eps > 0.0
