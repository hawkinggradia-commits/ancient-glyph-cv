import os
import cv2
import uuid
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from src.preprocess import clean_stone_inscription
from src.segmentation import segment_glyphs
from src.clustering import FeatureExtractor, evaluate_and_cluster

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/outputs'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16 MB limit
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize the heavy ML model once at startup to keep the API fast
extractor = FeatureExtractor(batch_size=32)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/process', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
        
    filename = secure_filename(file.filename)
    job_id = str(uuid.uuid4())[:8]
    raw_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}_raw_{filename}")
    file.save(raw_path)
    
    try:
        # 1. Preprocessing
        binarized = clean_stone_inscription(raw_path)
        bin_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}_bin.png")
        cv2.imwrite(bin_path, binarized)
        
        # 2. Segmentation
        glyphs = segment_glyphs(binarized, min_area=50, max_area=5000)
        
        if not glyphs:
            return jsonify({
                'raw_image': raw_path,
                'binarized_image': bin_path,
                'message': 'No valid glyphs detected based on current heuristics.',
                'clusters': []
            }), 200
            
        # 3. Embedding Extraction
        features = extractor.extract(glyphs)
        
        # 4. Clustering
        eps_search_range = np.arange(1.0, 15.0, 0.5)
        labels, optimal_eps = evaluate_and_cluster(features, eps_range=eps_search_range, min_samples=2)
        
        # Group and save the individual crops
        clusters_data = {}
        for i, (glyph, label) in enumerate(zip(glyphs, labels)):
            # convert label int to string for JSON keys
            c_name = "noise" if label == -1 else f"cluster_{label}"
            
            glyph_filename = f"{job_id}_glyph_{i}.png"
            glyph_path = os.path.join(app.config['UPLOAD_FOLDER'], glyph_filename)
            cv2.imwrite(glyph_path, glyph)
            
            if c_name not in clusters_data:
                clusters_data[c_name] = []
            clusters_data[c_name].append(glyph_path)
            
        # Format the response so the frontend can iterate it nicely
        formatted_clusters = [{"name": k, "images": v} for k, v in clusters_data.items()]
        # Sort so clusters are ordered (noise at the end if possible)
        formatted_clusters.sort(key=lambda x: 999 if x['name'] == 'noise' else int(x['name'].split('_')[1]))

        return jsonify({
            'raw_image': raw_path,
            'binarized_image': bin_path,
            'num_glyphs': len(glyphs),
            'clusters': formatted_clusters
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
