
from flask import Flask, render_template, request, jsonify, send_from_directory
import json
import os
from pathlib import Path
from datetime import datetime

app = Flask(__name__)

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/videos')
def get_videos():
    # Get list of processed videos
    processed_dir = Path('../02_preprocessing/processed_videos')
    videos = []
    
    for video_file in processed_dir.rglob('*.mp4'):
        videos.append({
            'name': video_file.stem,
            'path': str(video_file.relative_to(processed_dir)),
            'class': video_file.parent.name
        })
    
    return jsonify(videos)

@app.route('/api/annotations', methods=['POST'])
def save_annotations():
    data = request.json
    video_name = data.get('video')
    annotations = data.get('annotations', [])
    
    # Save annotations to ground truth directory
    ground_truth_dir = Path('../ground_truth_json')
    ground_truth_dir.mkdir(exist_ok=True)
    
    annotation_file = ground_truth_dir / f"{video_name}_annotations.json"
    
    annotation_data = {
        'video': video_name,
        'annotations': annotations,
        'timestamp': datetime.now().isoformat(),
        'annotator': 'user'  # In real implementation, get from session
    }
    
    with open(annotation_file, 'w') as f:
        json.dump(annotation_data, f, indent=2)
    
    return jsonify({'status': 'success'})

@app.route('/api/progress')
def get_progress():
    # Calculate annotation progress
    ground_truth_dir = Path('../ground_truth_json')
    processed_dir = Path('../02_preprocessing/processed_videos')
    
    total_videos = len(list(processed_dir.rglob('*.mp4')))
    annotated_videos = len(list(ground_truth_dir.glob('*.json')))
    
    return jsonify({
        'total': total_videos,
        'annotated': annotated_videos,
        'progress': (annotated_videos / total_videos * 100) if total_videos > 0 else 0
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
