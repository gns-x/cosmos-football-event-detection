#!/usr/bin/env python3
"""
SFT Dataset Builder for Cosmos Football Video Analysis
Creates training and validation datasets in SFT format with (prompt, completion) pairs
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
import argparse

class SFTDatasetBuilder:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.processed_videos_dir = self.project_root / "02_preprocessing" / "processed_videos"
        self.ground_truth_dir = self.project_root / "03_annotation" / "ground_truth_json"
        self.dataset_dir = self.project_root / "04_dataset"
        
        # Dataset splits
        self.train_split = 0.8
        self.val_split = 0.2
        
        # Ensure dataset directory exists
        self.dataset_dir.mkdir(exist_ok=True)
    
    def load_annotations(self) -> List[Dict[str, Any]]:
        """Load all annotation files."""
        print("📖 Loading ground truth annotations...")
        
        annotations = []
        annotation_files = list(self.ground_truth_dir.glob("*.json"))
        
        if not annotation_files:
            print("⚠️  No annotation files found!")
            return annotations
        
        for annotation_file in annotation_files:
            try:
                with open(annotation_file, 'r') as f:
                    data = json.load(f)
                    
                    # Handle both formats: direct array or wrapped object
                    if isinstance(data, list):
                        # Direct array format (exact specification)
                        annotations.append({
                            "video_file": annotation_file.stem,
                            "annotations": data
                        })
                    elif isinstance(data, dict) and "annotations" in data:
                        # Wrapped format
                        annotations.append(data)
                    else:
                        print(f"  ⚠️  Unknown format in {annotation_file.name}")
                        continue
                        
                print(f"  ✅ Loaded: {annotation_file.name}")
            except Exception as e:
                print(f"  ❌ Error loading {annotation_file.name}: {e}")
        
        print(f"📊 Total annotations loaded: {len(annotations)}")
        return annotations
    
    def find_video_files(self) -> Dict[str, List[Path]]:
        """Find all processed video files organized by class."""
        print("🔍 Finding processed video files...")
        
        video_files = {}
        video_count = 0
        
        for video_file in self.processed_videos_dir.rglob("*.mp4"):
            class_name = video_file.parent.name
            if class_name not in video_files:
                video_files[class_name] = []
            video_files[class_name].append(video_file)
            video_count += 1
        
        print(f"📹 Found {video_count} video files across {len(video_files)} classes")
        for class_name, files in video_files.items():
            print(f"  {class_name}: {len(files)} videos")
        
        return video_files
    
    def create_sft_prompt(self, video_path: Path, class_name: str) -> str:
        """Create the SFT prompt for a video."""
        prompt = """Analyze the following football clip. Identify all significant events including goals, cards, and shots. For each event, provide a detailed description including player, team, and jersey info, the event class, and precise start/end timestamps. Output *only* a valid JSON array."""
        return prompt
    
    def create_sft_completion(self, annotations: List[Dict[str, Any]]) -> str:
        """Create the SFT completion from annotations."""
        # Convert annotations to JSON string
        completion = json.dumps(annotations, ensure_ascii=False)
        return completion
    
    def create_sft_example(self, video_path: Path, annotations: List[Dict[str, Any]], class_name: str) -> Dict[str, Any]:
        """Create a single SFT training example."""
        # Create prompt
        prompt = self.create_sft_prompt(video_path, class_name)
        
        # Create completion
        completion = self.create_sft_completion(annotations)
        
        # Create SFT example
        sft_example = {
            "video": str(video_path.relative_to(self.project_root)),
            "prompt": prompt,
            "completion": completion
        }
        
        return sft_example
    
    def create_dataset_splits(self, annotations: List[Dict[str, Any]], video_files: Dict[str, List[Path]]) -> Tuple[List[Dict], List[Dict]]:
        """Create train/validation splits for SFT dataset."""
        print("📊 Creating SFT dataset splits...")
        
        # Group annotations by class
        class_annotations = {}
        for annotation in annotations:
            video_name = annotation.get('video_file', '')
            for video_file in self.processed_videos_dir.rglob(f"*{video_name}*"):
                if video_file.suffix == '.mp4':
                    class_name = video_file.parent.name
                    if class_name not in class_annotations:
                        class_annotations[class_name] = []
                    class_annotations[class_name].append({
                        'video_path': video_file,
                        'annotations': annotation.get('annotations', [])
                    })
                    break
        
        train_data = []
        val_data = []
        
        for class_name, class_data in class_annotations.items():
            # Shuffle annotations
            random.shuffle(class_data)
            
            # Calculate split sizes
            total = len(class_data)
            train_size = int(total * self.train_split)
            
            # Split data
            train_data.extend(class_data[:train_size])
            val_data.extend(class_data[train_size:])
            
            print(f"  {class_name}: {total} total")
            print(f"    Train: {train_size}, Val: {total - train_size}")
        
        return train_data, val_data
    
    def create_sft_dataset(self, train_data: List[Dict], val_data: List[Dict]):
        """Create SFT dataset files."""
        print("💾 Creating SFT dataset files...")
        
        # Create training dataset
        train_examples = []
        for item in train_data:
            video_path = item['video_path']
            annotations = item['annotations']
            class_name = video_path.parent.name
            
            sft_example = self.create_sft_example(video_path, annotations, class_name)
            train_examples.append(sft_example)
        
        # Create validation dataset
        val_examples = []
        for item in val_data:
            video_path = item['video_path']
            annotations = item['annotations']
            class_name = video_path.parent.name
            
            sft_example = self.create_sft_example(video_path, annotations, class_name)
            val_examples.append(sft_example)
        
        # Save to JSONL files
        self.save_jsonl(train_examples, self.dataset_dir / "train.jsonl")
        self.save_jsonl(val_examples, self.dataset_dir / "validation.jsonl")
        
        print(f"  ✅ Created: train.jsonl ({len(train_examples)} examples)")
        print(f"  ✅ Created: validation.jsonl ({len(val_examples)} examples)")
        
        return train_examples, val_examples
    
    def save_jsonl(self, data: List[Dict], filepath: Path):
        """Save data to JSONL file."""
        with open(filepath, 'w') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    def create_dataset_metadata(self, train_examples: List[Dict], val_examples: List[Dict]):
        """Create dataset metadata."""
        print("📝 Creating dataset metadata...")
        
        # Count classes
        classes = set()
        for example in train_examples + val_examples:
            # Extract class from video path
            video_path = Path(example['video'])
            if 'processed_videos' in str(video_path):
                class_name = video_path.parent.name
                classes.add(class_name)
        
        metadata = {
            "dataset_name": "Cosmos Football Video SFT Dataset",
            "version": "1.0.0",
            "created_date": datetime.now().isoformat(),
            "format": "SFT (Supervised Fine-Tuning)",
            "splits": {
                "train": len(train_examples),
                "validation": len(val_examples)
            },
            "classes": list(classes),
            "total_examples": len(train_examples) + len(val_examples),
            "sft_format": {
                "prompt": "Analyze the following football clip. Identify all significant events including goals, cards, and shots. For each event, provide a detailed description including player, team, and jersey info, the event class, and precise start/end timestamps. Output *only* a valid JSON array.",
                "completion_format": "JSON array of event annotations",
                "video_specs": {
                    "fps": 4,
                    "max_duration": 30,
                    "resolution": "720x480"
                }
            }
        }
        
        metadata_file = self.dataset_dir / "sft_dataset_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  ✅ Created: {metadata_file}")
    
    def create_dataset_summary(self, train_examples: List[Dict], val_examples: List[Dict]):
        """Create a summary of the SFT dataset."""
        print("📊 Creating SFT dataset summary...")
        
        # Count examples in each split
        train_count = len(train_examples)
        val_count = len(val_examples)
        
        # Count classes
        classes = set()
        for example in train_examples + val_examples:
            video_path = Path(example['video'])
            if 'processed_videos' in str(video_path):
                class_name = video_path.parent.name
                classes.add(class_name)
        
        summary = f"""
# 🏆 Cosmos Football Video SFT Dataset Summary

## 📊 Dataset Statistics
- **Total Examples**: {train_count + val_count}
- **Training**: {train_count} examples ({train_count/(train_count + val_count)*100:.1f}%)
- **Validation**: {val_count} examples ({val_count/(train_count + val_count)*100:.1f}%)

## 🏷️ Classes ({len(classes)})
{chr(10).join(f"- {cls}" for cls in sorted(classes))}

## 📁 Files
- `train.jsonl` - Training data in SFT format
- `validation.jsonl` - Validation data in SFT format
- `sft_dataset_metadata.json` - Dataset metadata

## 🎯 SFT Format
- **Prompt**: "Analyze the following football clip. Identify all significant events including goals, cards, and shots. For each event, provide a detailed description including player, team, and jersey info, the event class, and precise start/end timestamps. Output *only* a valid JSON array."
- **Completion**: JSON array of event annotations
- **Video FPS**: 4 (required by Cosmos-Reason1-7B)
- **Max Duration**: 30 seconds
- **Resolution**: 720x480

## 📝 Example SFT Entry
```json
{{
  "video": "02_preprocessing/processed_videos/goal/goal_sample_1_processed.mp4",
  "prompt": "Analyze the following football clip. Identify all significant events including goals, cards, and shots. For each event, provide a detailed description including player, team, and jersey info, the event class, and precise start/end timestamps. Output *only* a valid JSON array.",
  "completion": "[{{\\"description\\": \\"Player scores a goal in the goal video. The ball crosses the goal line and the referee signals a goal.\\", \\"start_time\\": \\"0:00:05\\", \\"end_time\\": \\"0:00:15\\", \\"event\\": \\"Goal\\"}}]"
}}
```

## 🚀 Next Steps
1. Review the SFT dataset format
2. Start training with `05_training/fine_tune.py`
3. Evaluate with `06_evaluation/evaluate.py`
"""
        
        summary_file = self.dataset_dir / "SFT_DATASET_README.md"
        with open(summary_file, 'w') as f:
            f.write(summary)
        
        print(f"  ✅ Created: {summary_file}")
    
    def build_sft_dataset(self):
        """Build the complete SFT dataset."""
        print("🏗️  Building Cosmos Football Video SFT Dataset")
        print("=" * 60)
        
        # Load annotations
        annotations = self.load_annotations()
        if not annotations:
            print("❌ No annotations found. Please run the annotation tool first.")
            return
        
        # Find video files
        video_files = self.find_video_files()
        
        # Create splits
        train_data, val_data = self.create_dataset_splits(annotations, video_files)
        
        # Create SFT dataset
        train_examples, val_examples = self.create_sft_dataset(train_data, val_data)
        
        # Create metadata
        self.create_dataset_metadata(train_examples, val_examples)
        
        # Create summary
        self.create_dataset_summary(train_examples, val_examples)
        
        print("=" * 60)
        print("✅ SFT dataset building completed!")
        print("")
        print("📁 SFT dataset files created in: 04_dataset/")
        print("🚀 Ready for Cosmos fine-tuning!")

def main():
    parser = argparse.ArgumentParser(description="Build SFT Dataset for Cosmos Football Video Analysis")
    parser.add_argument("--project-root", default="/Users/Genesis/Desktop/upwork/Nvidia-AI/project-cosmos-football", 
                       help="Project root directory")
    parser.add_argument("--train-split", type=float, default=0.8, 
                       help="Training split ratio")
    parser.add_argument("--val-split", type=float, default=0.2, 
                       help="Validation split ratio")
    
    args = parser.parse_args()
    
    # Validate splits
    if abs(args.train_split + args.val_split - 1.0) > 1e-6:
        print("❌ Error: Train + Val splits must equal 1.0")
        return
    
    builder = SFTDatasetBuilder(args.project_root)
    builder.train_split = args.train_split
    builder.val_split = args.val_split
    
    builder.build_sft_dataset()

if __name__ == "__main__":
    main()
