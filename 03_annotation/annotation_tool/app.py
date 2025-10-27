
from flask import Flask, render_template, request, jsonify, send_from_directory
import json
import os
from pathlib import Path
from datetime import datetime

app = Flask(__name__)

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

# The 8 specific football event classes from Task.md
EVENT_CLASSES = [
    "penalty_shot",
    "goal", 
    "goal_line_event",
    "woodworks",
    "shot_on_target",
    "red_card",
    "yellow_card",
    "hat_trick"
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/videos')
def get_videos():
    # Get list of raw videos from the 8 event classes
    # Use absolute path to avoid directory issues
    project_root = Path(__file__).parent.parent.parent
    raw_videos_dir = project_root / "01_data_collection" / "raw_videos"
    videos = []
    
    print(f"Looking for videos in: {raw_videos_dir.absolute()}")
    print(f"Directory exists: {raw_videos_dir.exists()}")
    
    for event_class in EVENT_CLASSES:
        class_dir = raw_videos_dir / event_class
        print(f"Checking class: {event_class} in {class_dir}")
        if class_dir.exists():
            video_files = list(class_dir.glob('*.mp4'))
            print(f"Found {len(video_files)} videos in {event_class}")
            for video_file in video_files:
                videos.append({
                    'name': video_file.stem,
                    'path': str(video_file.relative_to(raw_videos_dir)),
                    'class': event_class,
                    'full_path': str(video_file)
                })
        else:
            print(f"Class directory not found: {class_dir}")
    
    print(f"Total videos found: {len(videos)}")
    return jsonify(videos)

@app.route('/api/event-classes')
def get_event_classes():
    """Get the list of 8 specific event classes."""
    return jsonify(EVENT_CLASSES)

@app.route('/api/annotations', methods=['POST'])
def save_annotations():
    data = request.json
    video_name = data.get('video')
    annotations = data.get('annotations', [])
    event_class = data.get('event_class', 'unknown')
    
    # Save annotations to ground truth directory
    ground_truth_dir = Path('../ground_truth_json')
    ground_truth_dir.mkdir(exist_ok=True)
    
    annotation_file = ground_truth_dir / f"{video_name}_annotations.json"
    
    annotation_data = {
        'video': video_name,
        'event_class': event_class,
        'annotations': annotations,
        'timestamp': datetime.now().isoformat(),
        'annotator': 'user',  # In real implementation, get from session
        'data_format': {
            'description': 'Moment description including player, team names and jersey numbers',
            'start_time': 'Start timestamp of the moment',
            'end_time': 'End timestamp of the moment', 
            'event': 'Event type (one of the 8 classes)'
        }
    }
    
    with open(annotation_file, 'w') as f:
        json.dump(annotation_data, f, indent=2)
    
    return jsonify({'status': 'success'})

@app.route('/api/progress')
def get_progress():
    # Calculate annotation progress for the 8 event classes
    project_root = Path(__file__).parent.parent.parent
    ground_truth_dir = project_root / "03_annotation" / "ground_truth_json"
    raw_videos_dir = project_root / "01_data_collection" / "raw_videos"
    
    total_videos = 0
    annotated_videos = len(list(ground_truth_dir.glob('*.json')))
    
    # Count total videos in the 8 event classes
    for event_class in EVENT_CLASSES:
        class_dir = raw_videos_dir / event_class
        if class_dir.exists():
            total_videos += len(list(class_dir.glob('*.mp4')))
    
    progress = (annotated_videos / total_videos * 100) if total_videos > 0 else 0
    
    return jsonify({
        'total': total_videos,
        'annotated': annotated_videos,
        'progress': progress,
        'event_classes': EVENT_CLASSES
    })

@app.route('/api/video-stats')
def get_video_stats():
    """Get statistics about videos in each event class."""
    project_root = Path(__file__).parent.parent.parent
    raw_videos_dir = project_root / "01_data_collection" / "raw_videos"
    stats = {}
    
    for event_class in EVENT_CLASSES:
        class_dir = raw_videos_dir / event_class
        if class_dir.exists():
            video_count = len(list(class_dir.glob('*.mp4')))
            stats[event_class] = {
                'count': video_count,
                'videos': [f.stem for f in class_dir.glob('*.mp4')]
            }
        else:
            stats[event_class] = {'count': 0, 'videos': []}
    
    return jsonify(stats)

@app.route('/api/video/<path:video_path>')
def serve_video(video_path):
    """Serve video files."""
    project_root = Path(__file__).parent.parent.parent
    raw_videos_dir = project_root / "01_data_collection" / "raw_videos"
    video_file = raw_videos_dir / video_path
    
    if video_file.exists() and video_file.suffix == '.mp4':
        return send_from_directory(video_file.parent, video_file.name)
    else:
        return "Video not found", 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
