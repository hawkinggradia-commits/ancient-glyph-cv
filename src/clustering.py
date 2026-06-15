import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
import logging

logger = logging.getLogger(__name__)

class FeatureExtractor:
    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pre-trained ResNet18, strip classification head
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract(self, glyphs: list[np.ndarray]) -> np.ndarray:
        features = []
        with torch.no_grad():
            for i in range(0, len(glyphs), self.batch_size):
                batch_glyphs = glyphs[i:i + self.batch_size]
                tensors = torch.stack([self.transform(g) for g in batch_glyphs]).to(self.device)
                embeddings = self.model(tensors)
                features.extend(embeddings.view(embeddings.size(0), -1).cpu().numpy())
        return np.array(features)

def evaluate_and_cluster(features: np.ndarray, eps_range: np.ndarray = np.arange(1.0, 15.0, 0.5), min_samples: int = 2) -> tuple[np.ndarray, float]:
    """PCA Dimensionality Reduction + DBSCAN Clustering."""
    if len(features) < min_samples:
        return np.zeros(len(features), dtype=int), 0.0
        
    n_samples = len(features)
    n_components = min(n_samples, 50)
    
    if n_components > 1:
        pca = PCA(n_components=n_components, random_state=42)
        features_reduced = pca.fit_transform(features)
    else:
        features_reduced = features
        
    best_eps = eps_range[0]
    best_score = -1 
    best_labels = None
    
    for eps in eps_range:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(features_reduced)
        
        unique_labels = set(labels) - {-1}
        
        if len(unique_labels) > 1:
            sil_score = silhouette_score(features_reduced, labels)
            if sil_score > best_score:
                best_score = sil_score
                best_eps = eps
                best_labels = labels
            
    if best_labels is None:
        best_eps = eps_range[len(eps_range)//2]
        dbscan = DBSCAN(eps=best_eps, min_samples=min_samples)
        best_labels = dbscan.fit_predict(features_reduced)
        
    return best_labels, best_eps
