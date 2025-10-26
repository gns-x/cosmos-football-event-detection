#!/usr/bin/env python3
"""
Football Video Annotation Tool Setup
Creates annotation interface for football video classification
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Any

class FootballAnnotationSetup:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.annotation_dir = self.project_root / "03_annotation"
        self.tool_dir = self.annotation_dir / "annotation_tool"
        self.ground_truth_dir = self.annotation_dir / "ground_truth_json"
        
    def create_directories(self):
        """Create necessary directories for annotation."""
        print("📁 Creating annotation directories...")
        
        directories = [
            self.tool_dir,
            self.ground_truth_dir,
            self.tool_dir / "static",
            self.tool_dir / "templates",
            self.tool_dir / "data"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"  ✅ Created: {directory}")
    
    def create_annotation_config(self):
        """Create annotation configuration file."""
        print("⚙️  Creating annotation configuration...")
        
        config = {
            "project_name": "Cosmos Football Video Analysis",
            "version": "1.0.0",
            "classes": [
                {
                    "id": "penalty_shot",
                    "name": "Penalty Shot",
                    "description": "Player taking a penalty kick",
                    "color": "#FF6B6B"
                },
                {
                    "id": "goal",
                    "name": "Goal",
                    "description": "Ball crossing the goal line",
                    "color": "#4ECDC4"
                },
                {
                    "id": "red_card",
                    "name": "Red Card",
                    "description": "Referee showing red card",
                    "color": "#FF0000"
                },
                {
                    "id": "yellow_card",
                    "name": "Yellow Card",
                    "description": "Referee showing yellow card",
                    "color": "#FFFF00"
                },
                {
                    "id": "corner_kick",
                    "name": "Corner Kick",
                    "description": "Corner kick being taken",
                    "color": "#45B7D1"
                },
                {
                    "id": "free_kick",
                    "name": "Free Kick",
                    "description": "Free kick being taken",
                    "color": "#96CEB4"
                },
                {
                    "id": "throw_in",
                    "name": "Throw In",
                    "description": "Player taking throw-in",
                    "color": "#FFEAA7"
                },
                {
                    "id": "offside",
                    "name": "Offside",
                    "description": "Offside decision or situation",
                    "color": "#DDA0DD"
                }
            ],
            "annotation_settings": {
                "video_fps": 4,
                "max_video_duration": 30,
                "min_annotation_duration": 2,
                "max_annotation_duration": 10
            },
            "export_formats": ["json", "csv", "yolo"],
            "quality_control": {
                "min_annotations_per_video": 1,
                "max_annotations_per_video": 5,
                "require_confidence_score": True
            }
        }
        
        config_file = self.tool_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"  ✅ Created: {config_file}")
    
    def create_annotation_interface(self):
        """Create HTML annotation interface."""
        print("🖥️  Creating annotation interface...")
        
        html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Football Video Annotation Tool</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }
        .video-container {
            position: relative;
            background: #000;
            text-align: center;
        }
        video {
            width: 100%;
            max-width: 800px;
            height: auto;
        }
        .controls {
            padding: 20px;
            background: #f8f9fa;
            border-bottom: 1px solid #dee2e6;
        }
        .class-buttons {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
            margin-bottom: 20px;
        }
        .class-btn {
            padding: 15px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.3s ease;
            color: white;
        }
        .class-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        .class-btn.active {
            transform: scale(1.05);
            box-shadow: 0 6px 12px rgba(0,0,0,0.3);
        }
        .annotation-panel {
            padding: 20px;
        }
        .annotation-item {
            background: #e9ecef;
            padding: 15px;
            margin: 10px 0;
            border-radius: 8px;
            border-left: 4px solid #007bff;
        }
        .progress-bar {
            width: 100%;
            height: 20px;
            background: #e9ecef;
            border-radius: 10px;
            overflow: hidden;
            margin: 20px 0;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #28a745, #20c997);
            transition: width 0.3s ease;
        }
        .btn {
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
            margin: 5px;
        }
        .btn-primary {
            background: #007bff;
            color: white;
        }
        .btn-success {
            background: #28a745;
            color: white;
        }
        .btn-danger {
            background: #dc3545;
            color: white;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>⚽ Football Video Annotation Tool</h1>
            <p>Annotate football videos for Cosmos AI training</p>
        </div>
        
        <div class="video-container">
            <video id="videoPlayer" controls>
                <source src="" type="video/mp4">
                Your browser does not support the video tag.
            </video>
        </div>
        
        <div class="controls">
            <div class="progress-bar">
                <div class="progress-fill" id="progressBar" style="width: 0%"></div>
            </div>
            <p>Progress: <span id="progressText">0 / 0</span></p>
        </div>
        
        <div class="annotation-panel">
            <h3>Select Action Class:</h3>
            <div class="class-buttons" id="classButtons">
                <!-- Class buttons will be generated by JavaScript -->
            </div>
            
            <div id="annotationsList">
                <h3>Current Annotations:</h3>
                <div id="annotations"></div>
            </div>
            
            <div style="text-align: center; margin-top: 20px;">
                <button class="btn btn-primary" onclick="saveAnnotation()">Save Annotation</button>
                <button class="btn btn-success" onclick="nextVideo()">Next Video</button>
                <button class="btn btn-danger" onclick="skipVideo()">Skip Video</button>
            </div>
        </div>
    </div>

    <script>
        // Annotation tool JavaScript
        let currentVideo = 0;
        let videos = [];
        let annotations = [];
        let selectedClass = null;
        
        // Load configuration
        fetch('config.json')
            .then(response => response.json())
            .then(config => {
                initializeTool(config);
            });
        
        function initializeTool(config) {
            // Create class buttons
            const classButtons = document.getElementById('classButtons');
            config.classes.forEach(classItem => {
                const button = document.createElement('button');
                button.className = 'class-btn';
                button.style.backgroundColor = classItem.color;
                button.textContent = classItem.name;
                button.onclick = () => selectClass(classItem.id, button);
                classButtons.appendChild(button);
            });
            
            // Load video list
            loadVideoList();
        }
        
        function loadVideoList() {
            // This would typically load from a server
            // For now, we'll use placeholder data
            videos = [
                { path: 'data/video1.mp4', name: 'Video 1' },
                { path: 'data/video2.mp4', name: 'Video 2' },
                { path: 'data/video3.mp4', name: 'Video 3' }
            ];
            
            loadCurrentVideo();
        }
        
        function loadCurrentVideo() {
            if (currentVideo < videos.length) {
                const video = videos[currentVideo];
                document.getElementById('videoPlayer').src = video.path;
                updateProgress();
            } else {
                alert('All videos annotated!');
            }
        }
        
        function selectClass(classId, button) {
            // Remove active class from all buttons
            document.querySelectorAll('.class-btn').forEach(btn => {
                btn.classList.remove('active');
            });
            
            // Add active class to selected button
            button.classList.add('active');
            selectedClass = classId;
        }
        
        function saveAnnotation() {
            if (!selectedClass) {
                alert('Please select a class first!');
                return;
            }
            
            const video = videos[currentVideo];
            const annotation = {
                video: video.name,
                class: selectedClass,
                timestamp: new Date().toISOString(),
                confidence: 1.0
            };
            
            annotations.push(annotation);
            updateAnnotationsList();
            
            // Reset selection
            selectedClass = null;
            document.querySelectorAll('.class-btn').forEach(btn => {
                btn.classList.remove('active');
            });
        }
        
        function updateAnnotationsList() {
            const container = document.getElementById('annotations');
            container.innerHTML = '';
            
            annotations.forEach((annotation, index) => {
                const div = document.createElement('div');
                div.className = 'annotation-item';
                div.innerHTML = `
                    <strong>${annotation.class}</strong> - ${annotation.timestamp}
                    <button onclick="removeAnnotation(${index})" style="float: right; background: #dc3545; color: white; border: none; padding: 5px 10px; border-radius: 3px; cursor: pointer;">Remove</button>
                `;
                container.appendChild(div);
            });
        }
        
        function removeAnnotation(index) {
            annotations.splice(index, 1);
            updateAnnotationsList();
        }
        
        function nextVideo() {
            // Save current annotations
            saveCurrentAnnotations();
            
            // Move to next video
            currentVideo++;
            annotations = [];
            loadCurrentVideo();
        }
        
        function skipVideo() {
            currentVideo++;
            annotations = [];
            loadCurrentVideo();
        }
        
        function saveCurrentAnnotations() {
            if (annotations.length > 0) {
                const video = videos[currentVideo];
                const filename = `ground_truth_${video.name.replace(/\s+/g, '_')}.json`;
                
                // In a real implementation, this would save to server
                console.log('Saving annotations:', annotations);
                console.log('Filename:', filename);
            }
        }
        
        function updateProgress() {
            const progress = ((currentVideo + 1) / videos.length) * 100;
            document.getElementById('progressBar').style.width = progress + '%';
            document.getElementById('progressText').textContent = `${currentVideo + 1} / ${videos.length}`;
        }
    </script>
</body>
</html>
"""
        
        interface_file = self.tool_dir / "templates" / "index.html"
        with open(interface_file, 'w') as f:
            f.write(html_content)
        
        print(f"  ✅ Created: {interface_file}")
    
    def create_annotation_server(self):
        """Create Flask server for annotation tool."""
        print("🌐 Creating annotation server...")
        
        server_code = """
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
"""
        
        server_file = self.tool_dir / "app.py"
        with open(server_file, 'w') as f:
            f.write(server_code)
        
        print(f"  ✅ Created: {server_file}")
    
    def create_requirements(self):
        """Create requirements file for annotation tool."""
        print("📦 Creating requirements file...")
        
        requirements = [
            "flask>=2.0.0",
            "flask-cors>=3.0.0",
            "opencv-python>=4.8.0",
            "pillow>=9.5.0",
            "numpy>=1.24.0"
        ]
        
        requirements_file = self.tool_dir / "requirements.txt"
        with open(requirements_file, 'w') as f:
            f.write('\n'.join(requirements))
        
        print(f"  ✅ Created: {requirements_file}")
    
    def create_launch_script(self):
        """Create script to launch annotation tool."""
        print("🚀 Creating launch script...")
        
        launch_script = """#!/bin/bash

# Football Video Annotation Tool Launcher
echo "🚀 Starting Football Video Annotation Tool..."

# Check if Flask is installed
if ! python3 -c "import flask" 2>/dev/null; then
    echo "📦 Installing Flask..."
    pip install flask flask-cors
fi

# Start the annotation server
echo "🌐 Starting annotation server on http://localhost:5000"
echo "📝 Open your browser and go to: http://localhost:5000"
echo "⏹️  Press Ctrl+C to stop the server"
echo ""

cd "$(dirname "$0")"
python3 app.py
"""
        
        launch_file = self.tool_dir / "launch.sh"
        with open(launch_file, 'w') as f:
            f.write(launch_script)
        
        # Make executable
        os.chmod(launch_file, 0o755)
        
        print(f"  ✅ Created: {launch_file}")
    
    def setup_annotation_tool(self):
        """Set up the complete annotation tool."""
        print("🔧 Setting up Football Video Annotation Tool...")
        print("=" * 50)
        
        self.create_directories()
        self.create_annotation_config()
        self.create_annotation_interface()
        self.create_annotation_server()
        self.create_requirements()
        self.create_launch_script()
        
        print("=" * 50)
        print("✅ Annotation tool setup completed!")
        print("")
        print("🚀 To start the annotation tool:")
        print("   cd 03_annotation/annotation_tool")
        print("   ./launch.sh")
        print("")
        print("🌐 Then open your browser to: http://localhost:5000")

def main():
    project_root = "/Users/Genesis/Desktop/upwork/Nvidia-AI/project-cosmos-football"
    setup = FootballAnnotationSetup(project_root)
    setup.setup_annotation_tool()

if __name__ == "__main__":
    main()
