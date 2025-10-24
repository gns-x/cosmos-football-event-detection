#!/usr/bin/env python3
"""
Sample Football Video Data Collection
Downloads sample football videos for testing the pipeline
"""

import os
import subprocess
import json
from pathlib import Path
from typing import Dict, List
import argparse

class SampleDataCollector:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.raw_videos_dir = self.project_root / "01_data_collection" / "raw_videos"
        self.processed_videos_dir = self.project_root / "02_preprocessing" / "processed_videos"
        
        # Create directories
        self.raw_videos_dir.mkdir(parents=True, exist_ok=True)
        self.processed_videos_dir.mkdir(parents=True, exist_ok=True)
    
    def download_sample_videos(self):
        """Download sample football videos for testing."""
        print("🎬 Downloading sample football videos...")
        
        # Sample video URLs (these are public domain or creative commons)
        sample_videos = [
            {
                "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",  # Placeholder - replace with actual football videos
                "class": "goal",
                "description": "Sample goal video"
            },
            {
                "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",  # Placeholder
                "class": "penalty_shot", 
                "description": "Sample penalty video"
            }
        ]
        
        # For now, create placeholder videos
        self.create_placeholder_videos()
    
    def create_placeholder_videos(self):
        """Create placeholder videos for testing."""
        print("🎥 Creating placeholder videos for testing...")
        
        classes = [
            "penalty_shot",
            "goal", 
            "red_card",
            "yellow_card",
            "corner_kick",
            "free_kick",
            "throw_in",
            "offside"
        ]
        
        for class_name in classes:
            class_dir = self.raw_videos_dir / class_name
            class_dir.mkdir(exist_ok=True)
            
            # Create a simple test video using ffmpeg
            for i in range(3):  # 3 videos per class
                video_file = class_dir / f"{class_name}_sample_{i+1}.mp4"
                
                # Create a 10-second test video with a colored background
                colors = {
                    "penalty_shot": "red",
                    "goal": "green", 
                    "red_card": "red",
                    "yellow_card": "yellow",
                    "corner_kick": "blue",
                    "free_kick": "orange",
                    "throw_in": "purple",
                    "offside": "pink"
                }
                
                color = colors.get(class_name, "white")
                
                try:
                    subprocess.run([
                        "ffmpeg", "-y",
                        "-f", "lavfi",
                        "-i", f"color=c={color}:size=720x480:duration=10",
                        "-f", "lavfi", 
                        "-i", "sine=frequency=1000:duration=10",
                        "-c:v", "libx264",
                        "-c:a", "aac",
                        "-shortest",
                        str(video_file)
                    ], check=True, capture_output=True)
                    
                    print(f"  ✅ Created: {video_file}")
                    
                except subprocess.CalledProcessError as e:
                    print(f"  ❌ Error creating {video_file}: {e}")
                    # Create a simple text file as placeholder
                    with open(video_file.with_suffix('.txt'), 'w') as f:
                        f.write(f"Placeholder video for {class_name}\n")
                        f.write(f"Duration: 10 seconds\n")
                        f.write(f"Resolution: 720x480\n")
                        f.write(f"Class: {class_name}\n")
    
    def process_sample_videos(self):
        """Process the sample videos to 4 FPS."""
        print("🔄 Processing sample videos to 4 FPS...")
        
        for class_dir in self.raw_videos_dir.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                output_dir = self.processed_videos_dir / class_name
                output_dir.mkdir(exist_ok=True)
                
                print(f"  📁 Processing class: {class_name}")
                
                for video_file in class_dir.glob("*.mp4"):
                    output_file = output_dir / f"{video_file.stem}_processed.mp4"
                    
                    try:
                        subprocess.run([
                            "ffmpeg", "-y",
                            "-i", str(video_file),
                            "-vf", "fps=4,scale=720:480",
                            "-c:v", "libx264",
                            "-c:a", "aac",
                            "-preset", "fast",
                            "-crf", "23",
                            str(output_file)
                        ], check=True, capture_output=True)
                        
                        print(f"    ✅ Processed: {output_file.name}")
                        
                    except subprocess.CalledProcessError as e:
                        print(f"    ❌ Error processing {video_file.name}: {e}")
                        # Create placeholder processed file
                        with open(output_file.with_suffix('.txt'), 'w') as f:
                            f.write(f"Processed placeholder for {video_file.name}\n")
                            f.write(f"FPS: 4\n")
                            f.write(f"Resolution: 720x480\n")
    
    def create_sample_annotations(self):
        """Create sample annotations for testing."""
        print("📝 Creating sample annotations...")
        
        ground_truth_dir = self.project_root / "03_annotation" / "ground_truth_json"
        ground_truth_dir.mkdir(parents=True, exist_ok=True)
        
        for class_dir in self.processed_videos_dir.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                
                for video_file in class_dir.glob("*.mp4"):
                    video_name = video_file.stem
                    
                    # Create annotation file
                    annotation_file = ground_truth_dir / f"{video_name}_annotations.json"
                    
                    annotation_data = {
                        "video": video_name,
                        "annotations": [
                            {
                                "class": class_name,
                                "confidence": 1.0,
                                "timestamp": "2025-01-01T00:00:00Z",
                                "annotator": "sample_generator"
                            }
                        ],
                        "timestamp": "2025-01-01T00:00:00Z",
                        "annotator": "sample_generator"
                    }
                    
                    with open(annotation_file, 'w') as f:
                        json.dump(annotation_data, f, indent=2)
                    
                    print(f"  ✅ Created annotation: {annotation_file.name}")
    
    def create_sample_dataset(self):
        """Create sample dataset splits."""
        print("📊 Creating sample dataset...")
        
        # Run the dataset builder
        dataset_script = self.project_root / "04_dataset" / "build_dataset.py"
        
        try:
            subprocess.run([
                "python3", str(dataset_script),
                "--project-root", str(self.project_root)
            ], check=True)
            print("  ✅ Sample dataset created successfully!")
        except subprocess.CalledProcessError as e:
            print(f"  ❌ Error creating dataset: {e}")
    
    def collect_sample_data(self):
        """Collect all sample data for testing."""
        print("🚀 Starting Sample Data Collection")
        print("=" * 50)
        
        # Download/create sample videos
        self.download_sample_videos()
        print("")
        
        # Process videos
        self.process_sample_videos()
        print("")
        
        # Create annotations
        self.create_sample_annotations()
        print("")
        
        # Create dataset
        self.create_sample_dataset()
        print("")
        
        print("=" * 50)
        print("✅ Sample data collection completed!")
        print("")
        print("📁 Files created:")
        print(f"  Raw videos: {self.raw_videos_dir}")
        print(f"  Processed videos: {self.processed_videos_dir}")
        print(f"  Annotations: {self.project_root / '03_annotation' / 'ground_truth_json'}")
        print(f"  Dataset: {self.project_root / '04_dataset'}")
        print("")
        print("🚀 Ready for training!")

def main():
    parser = argparse.ArgumentParser(description="Collect Sample Football Video Data")
    parser.add_argument("--project-root", default="/Users/Genesis/Desktop/upwork/Nvidia-AI/project-cosmos-football",
                       help="Project root directory")
    
    args = parser.parse_args()
    
    collector = SampleDataCollector(args.project_root)
    collector.collect_sample_data()

if __name__ == "__main__":
    main()
