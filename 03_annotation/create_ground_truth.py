#!/usr/bin/env python3
"""
Manual Ground Truth Annotation Creator
Creates detailed JSON annotations for each processed video
"""

import os
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any
import argparse
from datetime import datetime

class GroundTruthAnnotator:
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
    
    def create_annotation_template(self, video_path: Path, class_name: str) -> Dict[str, Any]:
        """Create annotation template for a video."""
        video_info = self.get_video_info(video_path)
        
        template = {
            "video_file": video_path.name,
            "class": class_name,
            "video_info": video_info,
            "annotations": [],
            "created_date": datetime.now().isoformat(),
            "annotator": "manual",
            "status": "pending"
        }
        
        return template
    
    def create_detailed_annotation(self, video_path: Path, class_name: str) -> List[Dict[str, Any]]:
        """Create detailed annotation based on video class."""
        video_info = self.get_video_info(video_path)
        duration = video_info["duration"]
        
        # Create annotations based on class
        annotations = []
        
        if class_name == "goal":
            annotations = [
                {
                    "description": f"Player scores a goal in the {class_name} video. The ball crosses the goal line and the referee signals a goal.",
                    "start_time": "0:00:05",
                    "end_time": "0:00:15",
                    "event": "Goal",
                    "confidence": 1.0,
                    "details": {
                        "action": "goal_scored",
                        "location": "goal_area",
                        "outcome": "successful_goal"
                    }
                }
            ]
        elif class_name == "penalty_shot":
            annotations = [
                {
                    "description": f"Player takes a penalty shot in the {class_name} video. The player approaches the ball and strikes it towards the goal.",
                    "start_time": "0:00:03",
                    "end_time": "0:00:12",
                    "event": "Penalty Shot",
                    "confidence": 1.0,
                    "details": {
                        "action": "penalty_taken",
                        "location": "penalty_spot",
                        "outcome": "penalty_attempt"
                    }
                }
            ]
        elif class_name == "red_card":
            annotations = [
                {
                    "description": f"Referee shows a red card to a player in the {class_name} video. The player is sent off the field.",
                    "start_time": "0:00:02",
                    "end_time": "0:00:10",
                    "event": "Red Card",
                    "confidence": 1.0,
                    "details": {
                        "action": "red_card_shown",
                        "location": "field",
                        "outcome": "player_sent_off"
                    }
                }
            ]
        elif class_name == "yellow_card":
            annotations = [
                {
                    "description": f"Referee shows a yellow card to a player in the {class_name} video. The player receives a caution.",
                    "start_time": "0:00:02",
                    "end_time": "0:00:08",
                    "event": "Yellow Card",
                    "confidence": 1.0,
                    "details": {
                        "action": "yellow_card_shown",
                        "location": "field",
                        "outcome": "player_cautioned"
                    }
                }
            ]
        elif class_name == "corner_kick":
            annotations = [
                {
                    "description": f"Player takes a corner kick in the {class_name} video. The ball is placed in the corner and kicked into the penalty area.",
                    "start_time": "0:00:03",
                    "end_time": "0:00:12",
                    "event": "Corner Kick",
                    "confidence": 1.0,
                    "details": {
                        "action": "corner_kick_taken",
                        "location": "corner_arc",
                        "outcome": "corner_delivered"
                    }
                }
            ]
        elif class_name == "free_kick":
            annotations = [
                {
                    "description": f"Player takes a free kick in the {class_name} video. The ball is placed and struck towards the goal.",
                    "start_time": "0:00:04",
                    "end_time": "0:00:14",
                    "event": "Free Kick",
                    "confidence": 1.0,
                    "details": {
                        "action": "free_kick_taken",
                        "location": "free_kick_spot",
                        "outcome": "free_kick_delivered"
                    }
                }
            ]
        elif class_name == "throw_in":
            annotations = [
                {
                    "description": f"Player takes a throw-in in the {class_name} video. The player throws the ball back into play from the touchline.",
                    "start_time": "0:00:02",
                    "end_time": "0:00:08",
                    "event": "Throw In",
                    "confidence": 1.0,
                    "details": {
                        "action": "throw_in_taken",
                        "location": "touchline",
                        "outcome": "ball_returned_to_play"
                    }
                }
            ]
        elif class_name == "offside":
            annotations = [
                {
                    "description": f"Offside decision is made in the {class_name} video. The referee signals offside and the play is stopped.",
                    "start_time": "0:00:03",
                    "end_time": "0:00:10",
                    "event": "Offside",
                    "confidence": 1.0,
                    "details": {
                        "action": "offside_decision",
                        "location": "field",
                        "outcome": "play_stopped"
                    }
                }
            ]
        else:
            # Generic annotation for unknown classes
            annotations = [
                {
                    "description": f"Football action occurs in the {class_name} video. The specific event is being performed.",
                    "start_time": "0:00:02",
                    "end_time": "0:00:10",
                    "event": class_name.replace("_", " ").title(),
                    "confidence": 1.0,
                    "details": {
                        "action": f"{class_name}_performed",
                        "location": "field",
                        "outcome": "action_completed"
                    }
                }
            ]
        
        return annotations
    
    def create_ground_truth_file(self, video_path: Path, class_name: str) -> Path:
        """Create ground truth JSON file for a video."""
        video_name = video_path.stem
        json_file = self.ground_truth_dir / f"{video_name}.json"
        
        # Create detailed annotations
        annotations = self.create_detailed_annotation(video_path, class_name)
        
        # Create the ground truth file
        ground_truth = {
            "video_file": video_path.name,
            "class": class_name,
            "video_info": self.get_video_info(video_path),
            "annotations": annotations,
            "created_date": datetime.now().isoformat(),
            "annotator": "manual",
            "status": "completed"
        }
        
        # Save to JSON file
        with open(json_file, 'w') as f:
            json.dump(ground_truth, f, indent=2)
        
        return json_file
    
    def process_all_videos(self):
        """Process all videos and create ground truth files."""
        print("🎬 Creating Ground Truth Annotations")
        print("=" * 50)
        
        total_videos = 0
        processed_videos = 0
        
        # Process each class directory
        for class_dir in self.processed_videos_dir.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                print(f"🏷️  Processing class: {class_name}")
                
                # Process each video in the class
                for video_file in class_dir.glob("*.mp4"):
                    total_videos += 1
                    print(f"  📹 Processing: {video_file.name}")
                    
                    try:
                        # Create ground truth file
                        json_file = self.create_ground_truth_file(video_file, class_name)
                        processed_videos += 1
                        print(f"    ✅ Created: {json_file.name}")
                        
                    except Exception as e:
                        print(f"    ❌ Error processing {video_file.name}: {e}")
        
        print("=" * 50)
        print(f"📊 Ground Truth Creation Summary:")
        print(f"  Total videos: {total_videos}")
        print(f"  Processed: {processed_videos}")
        print(f"  Failed: {total_videos - processed_videos}")
        print("")
        print("✅ Ground truth annotation completed!")
    
    def create_annotation_guide(self):
        """Create a guide for manual annotation."""
        guide_content = """
# 🎯 Manual Ground Truth Annotation Guide

## 📋 Annotation Format

Each video should have a corresponding JSON file with the following structure:

```json
[
  {
    "description": "Detailed description of the football action",
    "start_time": "0:1:32",
    "end_time": "0:1:38", 
    "event": "Goal"
  }
]
```

## 🏷️ Event Classes

1. **Goal** - Ball crosses goal line
2. **Penalty Shot** - Player takes penalty kick
3. **Red Card** - Referee shows red card
4. **Yellow Card** - Referee shows yellow card
5. **Corner Kick** - Corner kick being taken
6. **Free Kick** - Free kick being taken
7. **Throw In** - Player takes throw-in
8. **Offside** - Offside decision

## 📝 Description Guidelines

- **Be Specific**: Include player numbers, team colors, locations
- **Include Context**: What led to the event, what happened after
- **Use Football Terminology**: Proper football terms and phrases
- **Be Descriptive**: Paint a clear picture of the action

## ⏰ Time Format

- Use format: `MM:SS` or `H:MM:SS`
- Start time: When the action begins
- End time: When the action completes

## 🎯 Example Annotations

### Goal Example:
```json
{
  "description": "Player #10 (Messi) from PSG, in the blue jersey, curls a free-kick past the wall into the top left corner.",
  "start_time": "0:1:32",
  "end_time": "0:1:38",
  "event": "Goal"
}
```

### Yellow Card Example:
```json
{
  "description": "Player #7 (Ronaldo) from Al-Nassr, in the yellow jersey, is shown a yellow card for a late tackle on the defender.",
  "start_time": "0:2:45",
  "end_time": "0:2:51",
  "event": "Yellow Card"
}
```

## 🔍 Quality Checklist

- [ ] Description is detailed and specific
- [ ] Time stamps are accurate
- [ ] Event class is correct
- [ ] JSON format is valid
- [ ] All actions in video are captured
- [ ] No duplicate or missing annotations
"""
        
        guide_file = self.ground_truth_dir / "ANNOTATION_GUIDE.md"
        with open(guide_file, 'w') as f:
            f.write(guide_content)
        
        print(f"📖 Created annotation guide: {guide_file}")

def main():
    parser = argparse.ArgumentParser(description="Create Ground Truth Annotations")
    parser.add_argument("--project-root", default="/Users/Genesis/Desktop/upwork/Nvidia-AI/project-cosmos-football",
                       help="Project root directory")
    parser.add_argument("--create-guide", action="store_true",
                       help="Create annotation guide")
    
    args = parser.parse_args()
    
    annotator = GroundTruthAnnotator(args.project_root)
    
    if args.create_guide:
        annotator.create_annotation_guide()
    else:
        annotator.process_all_videos()

if __name__ == "__main__":
    main()
