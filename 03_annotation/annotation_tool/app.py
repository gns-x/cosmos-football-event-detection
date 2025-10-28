
from flask import Flask, render_template, request, jsonify, send_from_directory
import json
import os
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime
from urllib.parse import unquote, quote

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
    
    print(f"[DEBUG] Looking for videos in: {raw_videos_dir.absolute()}")
    print(f"[DEBUG] Directory exists: {raw_videos_dir.exists()}")
    
    for event_class in EVENT_CLASSES:
        class_dir = raw_videos_dir / event_class
        print(f"[DEBUG] Checking class: {event_class} in {class_dir}")
        if class_dir.exists():
            video_files = list(class_dir.glob('*.mp4'))
            print(f"[DEBUG] Found {len(video_files)} videos in {event_class}")
            for video_file in video_files:
                relative_path = str(video_file.relative_to(raw_videos_dir))
                # Pre-encode URL path to preserve characters like '#', spaces, unicode etc.
                url_path = quote(relative_path, safe='/._-()')
                print(f"[DEBUG] Video: {video_file.name} -> {relative_path} | url_path={url_path}")
                videos.append({
                    'name': video_file.stem,
                    'path': relative_path,
                    'url_path': url_path,
                    'class': event_class,
                    'full_path': str(video_file)
                })
        else:
            print(f"[DEBUG] Class directory not found: {class_dir}")
    
    print(f"[DEBUG] Total videos found: {len(videos)}")
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

def detect_scene_changes(video_path: Path) -> list:
    """Detect scene changes using ffmpeg scene detection."""
    try:
        # Use ffmpeg to detect scene changes
        cmd = [
            'ffmpeg', '-i', str(video_path),
            '-vf', 'select=gt(scene\\,0.3)', '-vsync', 'vfr',
            '-f', 'null', '-'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        # Parse output to extract timestamps
        timestamps = []
        for line in result.stderr.split('\n'):
            if 'select:1' in line and 'pts_time:' in line:
                try:
                    # Extract timestamp from line like "select:1 n=123 pts_time:12.34"
                    parts = line.split('pts_time:')
                    if len(parts) > 1:
                        timestamp = float(parts[1].split()[0])
                        timestamps.append(timestamp)
                except (ValueError, IndexError):
                    continue
        
        return sorted(set(timestamps))
    except Exception as e:
        print(f"[DEBUG] Scene detection error: {e}")
        return []

def generate_draft_annotations(video_path: Path, event_class: str, scene_timestamps: list) -> list:
    """Generate draft annotations based on scene changes."""
    annotations = []
    
    for i, timestamp in enumerate(scene_timestamps):
        # Create annotation for each scene change
        annotation = {
            "description": f"Auto-detected {event_class.replace('_', ' ')} event at scene change",
            "start_time": format_time(timestamp),
            "end_time": format_time(min(timestamp + 3, get_video_duration(video_path))),  # 3 second duration
            "event": event_class,
            "confidence": 0.7,  # Medium confidence for auto-generated
            "auto_generated": True
        }
        annotations.append(annotation)
    
    return annotations

def format_time(seconds: float) -> str:
    """Format seconds to MM:SS format."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}:{secs:02d}"

def get_video_duration(video_path: Path) -> float:
    """Get video duration using ffprobe."""
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
            '-of', 'csv=p=0', str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return float(result.stdout.strip())
    except:
        return 300.0  # Default 5 minutes

@app.route('/api/auto-annotate', methods=['POST'])
def auto_annotate():
    """Auto-generate draft annotations for a video."""
    data = request.json
    video_name = data.get('video')
    event_class = data.get('event_class')
    
    if not video_name or not event_class:
        return jsonify({'error': 'Missing video or event_class'}), 400
    
    try:
        # Find the video file
        project_root = Path(__file__).parent.parent.parent
        raw_videos_dir = project_root / "01_data_collection" / "raw_videos"
        
        video_file = None
        for event_cls in EVENT_CLASSES:
            class_dir = raw_videos_dir / event_cls
            if class_dir.exists():
                for vf in class_dir.glob('*.mp4'):
                    if vf.stem == video_name:
                        video_file = vf
                        break
                if video_file:
                    break
        
        if not video_file:
            return jsonify({'error': 'Video file not found'}), 404
        
        print(f"[DEBUG] Auto-annotating: {video_file}")
        
        # Detect scene changes
        scene_timestamps = detect_scene_changes(video_file)
        print(f"[DEBUG] Found {len(scene_timestamps)} scene changes")
        
        # Generate draft annotations
        draft_annotations = generate_draft_annotations(video_file, event_class, scene_timestamps)
        
        return jsonify({
            'status': 'success',
            'annotations': draft_annotations,
            'scene_count': len(scene_timestamps)
        })
        
    except Exception as e:
        print(f"[DEBUG] Auto-annotation error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/video/<path:video_path>')
def serve_video(video_path):
    """Serve video files."""
    project_root = Path(__file__).parent.parent.parent
    raw_videos_dir = project_root / "01_data_collection" / "raw_videos"
    
    # URL decode the video path
    decoded_path = unquote(video_path)
    video_file = raw_videos_dir / decoded_path
    
    print(f"[DEBUG] Serving video: {video_path}")
    print(f"[DEBUG] Decoded path: {decoded_path}")
    print(f"[DEBUG] Full path: {video_file}")
    print(f"[DEBUG] Exists: {video_file.exists()}")
    print(f"[DEBUG] Suffix: {video_file.suffix}")
    
    if video_file.exists() and video_file.suffix == '.mp4':
        print(f"[DEBUG] Serving from: {video_file.parent}")
        return send_from_directory(str(video_file.parent), video_file.name)
    else:
        print(f"[DEBUG] Video not found or invalid")
        return f"Video not found: {video_path}", 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
