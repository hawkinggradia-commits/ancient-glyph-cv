import os
import cv2
import numpy as np
import logging
from src.preprocess import clean_stone_inscription
from src.segmentation import segment_glyphs
from src.clustering import FeatureExtractor, evaluate_and_cluster

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s] %(message)s')
logger = logging.getLogger("PipelineOrchestrator")

def save_clusters(glyphs: list[np.ndarray], source_files: list[str], labels: np.ndarray, output_dir: str = "data/clusters") -> None:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for i, (glyph, src, label) in enumerate(zip(glyphs, source_files, labels)):
        cluster_name = "noise" if label == -1 else f"cluster_{label}"
        cluster_dir = os.path.join(output_dir, cluster_name)
        if not os.path.exists(cluster_dir):
            os.makedirs(cluster_dir)
            
        src_basename = os.path.basename(src).split('.')[0]
        filename = os.path.join(cluster_dir, f"{src_basename}_glyph_{i}.png")
        cv2.imwrite(filename, glyph)
    
    unique_clusters = set(labels) - {-1}
    logger.info(f"Saved {len(glyphs)} total glyphs across {len(unique_clusters)} distinct characters in '{output_dir}/'")

def main(input_dir: str):
    logger.info(f"Systems Architecture Pipeline Initiated on directory: {input_dir}")
    
    all_glyphs = []
    all_source_files = []
    
    image_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        logger.error(f"No images found in {input_dir}.")
        return

    logger.info(f"Found {len(image_files)} fragments in the corpus.")
    
    # Optional: ensure processed directory exists
    os.makedirs("data/processed", exist_ok=True)
    
    for img_path in image_files:
        logger.info(f"Processing {os.path.basename(img_path)}...")
        
        # Phase 1: Preprocessing
        binarized = clean_stone_inscription(img_path)
        
        # Save processed intermediate for transparency
        cv2.imwrite(os.path.join("data/processed", f"bin_{os.path.basename(img_path)}"), binarized)
        
        # Phase 2: Segmentation
        glyphs = segment_glyphs(binarized)
        
        logger.info(f"  -> Extracted {len(glyphs)} potential glyph components.")
        all_glyphs.extend(glyphs)
        all_source_files.extend([img_path] * len(glyphs))

    if len(all_glyphs) == 0:
        logger.warning("No glyphs found in the corpus.")
        return
        
    logger.info(f"[Phase 3.1] Generating Latent Space Embeddings for {len(all_glyphs)} glyphs...")
    extractor = FeatureExtractor(batch_size=32)
    features = extractor.extract(all_glyphs)
    
    logger.info("[Phase 3.2] Cross-Fragment DBSCAN Clustering")
    eps_search_range = np.arange(1.0, 15.0, 0.5)
    labels, optimal_eps = evaluate_and_cluster(features, eps_range=eps_search_range, min_samples=2)
    
    logger.info("[Output] Saving Corpus Clusters...")
    save_clusters(all_glyphs, all_source_files, labels)
    logger.info("Pipeline complete.")

if __name__ == "__main__":
    import sys
    input_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    if os.path.exists(input_path):
        main(input_path)
    else:
        logger.error(f"'{input_path}' not found.")
