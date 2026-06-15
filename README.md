# Ancient Glyph Computer Vision Pipeline

This repository hosts a scalable, automated computer vision pipeline designed to restore, segment, and classify severely degraded glyphs from ancient, partially eroded Indian inscriptions (e.g., Ashokan Brahmi, Indus Valley symbols) sourced from archaeological catalogs.

## The Architecture
The system decouples the raw OpenCV scripts into a modular data pipeline, transitioning away from arbitrary matrix manipulation into a deterministic, mathematically verifiable Machine Learning workflow.

### Technical Directory Structure
- `data/raw/`: Original unedited inscription images.
- `data/processed/`: Segmented, binarized character crops and matrices.
- `data/clusters/`: Final output directories grouped by unsupervised DBSCAN clustering.
- `src/preprocess.py`: Implements Bilateral Filtering, CLAHE, and Adaptive Thresholding to kill rock grain noise while preserving etched contours.
- `src/segmentation.py`: Uses Connected Component Analysis and specific bounding-box heuristics (Area, Aspect Ratio constraints) to filter out physical rock fractures from actual language strokes.
- `src/clustering.py`: Leverages a pre-trained `ResNet18` to map the segmented spatial shapes into a dense 512D latent embedding. Reduces dimensionality via PCA, then automatically discovers the number of distinct characters ($K$) by computing the Silhouette Coefficient and running DBSCAN.

### Getting Started
1. Place raw `.jpg` images in `data/raw/`.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run the orchestrator: `python main.py data/raw`.
4. Review distinct matched characters inside `data/clusters/`.
