#!/usr/bin/env python3
"""
Interactive Manual Annotation Tool
Allows manual creation of detailed ground truth annotations
"""

import os
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any
import argparse
from datetime import datetime

class ManualAnnotator:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.processed_videos_dir = self.project_root / "02_preprocessing" / "processed_videos"
        self.ground_truth_dir = self.project_root / "03_annotation" / "ground_truth_json"
        
        # Create ground truth directory
        self.ground_truth_dir.mkdir(parents=True, exist_ok=True)
    
    def get_video_info(self, video_path: Path) -> Dict[str, Any]:
        """Get video information using ffprobe."""
        try:
            # Get duration
            duration_cmd = [
                "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                "-of", "csv=p=0", str(video_path)
            ]
            duration = float(subprocess.check_output(duration_cmd).decode().strip())
            
            # Get FPS
            fps_cmd = [
                "ffprobe", "-v", "quiet", "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", str(video_path)
            ]
            fps = subprocess.check_output(fps_cmd).decode().strip()
            
            return {
                "duration": duration,
                "fps": fps,
                "duration_formatted": f"{int(duration//60)}:{int(duration%60):02d}"
            }
        except Exception as e:
            print(f"❌ Error getting video info for {video_path}: {e}")
            return {"duration": 0, "fps": "unknown", "duration_formatted": "0:00"}
    
    def play_video(self, video_path: Path):
        """Play video for annotation."""
        print(f"🎬 Playing video: {video_path.name}")
        print(f"📁 Path: {video_path}")
        print("⏹️  Press 'q' to quit the video player")
        print("")
        
        try:
            # Try to play with system default player
            subprocess.run(["open", str(video_path)], check=True)
        except:
            try:
                # Fallback to ffplay
                subprocess.run(["ffplay", str(video_path)], check=True)
            except:
                print("⚠️  Could not play video automatically. Please open manually:")
                print(f"   {video_path}")
    
    def get_annotation_input(self, video_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get manual annotation input from user."""
        annotations = []
        
        print("📝 Manual Annotation Input")
        print("=" * 40)
        print(f"Video Duration: {video_info['duration_formatted']}")
        print("")
        
        while True:
            print("🎯 Add New Annotation:")
            print("")
            
            # Get description
            description = input("📝 Description (detailed): ").strip()
            if not description:
                break
            
            # Get start time
            start_time = input("⏰ Start time (MM:SS or H:MM:SS): ").strip()
            if not start_time:
                break
            
            # Get end time
            end_time = input("⏰ End time (MM:SS or H:MM:SS): ").strip()
            if not end_time:
                break
            
            # Get event type
            print("🏷️  Event types:")
            print("  1. Goal")
            print("  2. Penalty Shot")
            print("  3. Red Card")
            print("  4. Yellow Card")
            print("  5. Corner Kick")
            print("  6. Free Kick")
            print("  7. Throw In")
            print("  8. Offside")
            print("  9. Custom")
            
            event_choice = input("🎯 Event type (1-9): ").strip()
            
            event_map = {
                "1": "Goal",
                "2": "Penalty Shot",
                "3": "Red Card",
                "4": "Yellow Card",
                "5": "Corner Kick",
                "6": "Free Kick",
                "7": "Throw In",
                "8": "Offside"
            }
            
            if event_choice in event_map:
                event = event_map[event_choice]
            elif event_choice == "9":
                event = input("📝 Custom event name: ").strip()
            else:
                event = "Unknown"
            
            # Get confidence
            confidence = input("🎯 Confidence (0.0-1.0, default 1.0): ").strip()
            try:
                confidence = float(confidence) if confidence else 1.0
            except:
                confidence = 1.0
            
            # Create annotation
            annotation = {
                "description": description,
                "start_time": start_time,
                "end_time": end_time,
                "event": event,
                "confidence": confidence
            }
            
            annotations.append(annotation)
            print(f"✅ Added annotation: {event}")
            print("")
            
            # Ask for more annotations
            more = input("➕ Add another annotation? (y/n): ").strip().lower()
            if more != 'y':
                break
        
        return annotations
    
    def create_ground_truth_file(self, video_path: Path, class_name: str) -> Path:
        """Create ground truth JSON file for a video."""
        video_name = video_path.stem
        json_file = self.ground_truth_dir / f"{video_name}.json"
        
        # Get video info
        video_info = self.get_video_info(video_path)
        
        # Play video for annotation
        self.play_video(video_path)
        
        # Get manual annotations
        annotations = self.get_annotation_input(video_info)
        
        if not annotations:
            print("⚠️  No annotations provided. Creating empty file.")
            annotations = []
        
        # Create the ground truth file in the exact format specified
        ground_truth = annotations  # Use the exact format: array of annotation objects
        
        # Save to JSON file
        with open(json_file, 'w') as f:
            json.dump(ground_truth, f, indent=2)
        
        print(f"✅ Created ground truth file: {json_file}")
        return json_file
    
    def annotate_single_video(self, video_path: Path, class_name: str):
        """Annotate a single video."""
        print(f"🎬 Annotating: {video_path.name}")
        print(f"🏷️  Class: {class_name}")
        print("")
        
        try:
            json_file = self.create_ground_truth_file(video_path, class_name)
            print(f"✅ Annotation completed: {json_file}")
        except Exception as e:
            print(f"❌ Error annotating {video_path.name}: {e}")
    
    def annotate_all_videos(self):
        """Annotate all videos interactively."""
        print("🎬 Interactive Manual Annotation")
        print("=" * 50)
        
        total_videos = 0
        annotated_videos = 0
        
        # Process each class directory
        for class_dir in self.processed_videos_dir.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                print(f"🏷️  Processing class: {class_name}")
                
                # Process each video in the class
                for video_file in class_dir.glob("*.mp4"):
                    total_videos += 1
                    print(f"")
                    print(f"📹 Video {total_videos}: {video_file.name}")
                    
                    # Ask if user wants to annotate this video
                    annotate = input(f"🎯 Annotate this video? (y/n/skip): ").strip().lower()
                    
                    if annotate == 'y':
                        self.annotate_single_video(video_file, class_name)
                        annotated_videos += 1
                    elif annotate == 'skip':
                        print("⏭️  Skipping video")
                    else:
                        print("⏭️  Skipping video")
        
        print("=" * 50)
        print(f"📊 Annotation Summary:")
        print(f"  Total videos: {total_videos}")
        print(f"  Annotated: {annotated_videos}")
        print(f"  Skipped: {total_videos - annotated_videos}")
        print("")
        print("✅ Manual annotation completed!")

def main():
    parser = argparse.ArgumentParser(description="Interactive Manual Annotation Tool")
    parser.add_argument("--project-root", default=".",
                       help="Project root directory")
    parser.add_argument("--video", type=str,
                       help="Specific video file to annotate")
    parser.add_argument("--class-name", type=str, dest="class_name",
                       help="Class name for the video")
    
    args = parser.parse_args()
    
    annotator = ManualAnnotator(args.project_root)
    
    if args.video and args.class_name:
        # Annotate specific video
        video_path = Path(args.video)
        annotator.annotate_single_video(video_path, args.class_name)
    else:
        # Annotate all videos
        annotator.annotate_all_videos()

if __name__ == "__main__":
    main()
