#!/usr/bin/env python3
"""
Football Video Dataset Builder
Creates training, validation, and test datasets from annotated videos
"""

import os
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
import argparse

class FootballDatasetBuilder:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.processed_videos_dir = self.project_root / "02_preprocessing" / "processed_videos"
        self.ground_truth_dir = self.project_root / "03_annotation" / "ground_truth_json"
        self.dataset_dir = self.project_root / "04_dataset"
        
        # Dataset splits
        self.train_split = 0.7
        self.val_split = 0.15
        self.test_split = 0.15
        
        # Ensure dataset directory exists
        self.dataset_dir.mkdir(exist_ok=True)
    
    def load_annotations(self) -> List[Dict[str, Any]]:
        """Load all annotation files."""
        print("📖 Loading annotations...")
        
        annotations = []
        annotation_files = list(self.ground_truth_dir.glob("*.json"))
        
        if not annotation_files:
            print("⚠️  No annotation files found!")
            return annotations
        
        for annotation_file in annotation_files:
            try:
                with open(annotation_file, 'r') as f:
                    data = json.load(f)
                    annotations.append(data)
                print(f"  ✅ Loaded: {annotation_file.name}")
            except Exception as e:
                print(f"  ❌ Error loading {annotation_file.name}: {e}")
        
        print(f"📊 Total annotations loaded: {len(annotations)}")
        return annotations
    
    def find_video_files(self) -> Dict[str, List[Path]]:
        """Find all processed video files organized by class."""
        print("🔍 Finding video files...")
        
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
    
    def create_dataset_splits(self, annotations: List[Dict[str, Any]], video_files: Dict[str, List[Path]]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Create train/validation/test splits."""
        print("📊 Creating dataset splits...")
        
        # Group annotations by class
        class_annotations = {}
        for annotation in annotations:
            video_name = annotation.get('video', '')
            for ann in annotation.get('annotations', []):
                class_name = ann.get('class', 'unknown')
                if class_name not in class_annotations:
                    class_annotations[class_name] = []
                class_annotations[class_name].append(annotation)
        
        train_data = []
        val_data = []
        test_data = []
        
        for class_name, class_anns in class_annotations.items():
            # Shuffle annotations
            random.shuffle(class_anns)
            
            # Calculate split sizes
            total = len(class_anns)
            train_size = int(total * self.train_split)
            val_size = int(total * self.val_split)
            
            # Split data
            train_data.extend(class_anns[:train_size])
            val_data.extend(class_anns[train_size:train_size + val_size])
            test_data.extend(class_anns[train_size + val_size:])
            
            print(f"  {class_name}: {len(class_anns)} total")
            print(f"    Train: {train_size}, Val: {val_size}, Test: {total - train_size - val_size}")
        
        return train_data, val_data, test_data
    
    def create_cosmos_format(self, annotations: List[Dict[str, Any]], split_name: str) -> List[Dict[str, Any]]:
        """Convert annotations to Cosmos format."""
        print(f"🔄 Converting to Cosmos format for {split_name}...")
        
        cosmos_data = []
        
        for annotation in annotations:
            video_name = annotation.get('video', '')
            video_annotations = annotation.get('annotations', [])
            
            if not video_annotations:
                continue
            
            # Find corresponding video file
            video_file = None
            for video_path in self.processed_videos_dir.rglob(f"*{video_name}*"):
                if video_path.suffix == '.mp4':
                    video_file = video_path
                    break
            
            if not video_file:
                print(f"  ⚠️  Video file not found for: {video_name}")
                continue
            
            # Create Cosmos format entry
            for ann in video_annotations:
                cosmos_entry = {
                    "id": f"{video_name}_{ann.get('class', 'unknown')}_{len(cosmos_data)}",
                    "video_path": str(video_file.relative_to(self.project_root)),
                    "class": ann.get('class', 'unknown'),
                    "confidence": ann.get('confidence', 1.0),
                    "timestamp": ann.get('timestamp', datetime.now().isoformat()),
                    "annotator": ann.get('annotator', 'user'),
                    "metadata": {
                        "fps": 4,  # Cosmos requirement
                        "duration": 30,  # Max duration
                        "resolution": "720x480",
                        "split": split_name
                    }
                }
                cosmos_data.append(cosmos_entry)
        
        print(f"  ✅ Created {len(cosmos_data)} Cosmos format entries")
        return cosmos_data
    
    def save_dataset(self, train_data: List[Dict], val_data: List[Dict], test_data: List[Dict]):
        """Save dataset splits to JSONL files."""
        print("💾 Saving dataset splits...")
        
        # Convert to Cosmos format
        train_cosmos = self.create_cosmos_format(train_data, "train")
        val_cosmos = self.create_cosmos_format(val_data, "validation")
        test_cosmos = self.create_cosmos_format(test_data, "test")
        
        # Save to JSONL files
        self.save_jsonl(train_cosmos, self.dataset_dir / "train.jsonl")
        self.save_jsonl(val_cosmos, self.dataset_dir / "validation.jsonl")
        self.save_jsonl(test_cosmos, self.dataset_dir / "test.jsonl")
        
        # Create dataset metadata
        metadata = {
            "dataset_name": "Cosmos Football Video Analysis",
            "version": "1.0.0",
            "created_date": datetime.now().isoformat(),
            "splits": {
                "train": len(train_cosmos),
                "validation": len(val_cosmos),
                "test": len(test_cosmos)
            },
            "classes": list(set([item["class"] for item in train_cosmos + val_cosmos + test_cosmos])),
            "total_samples": len(train_cosmos) + len(val_cosmos) + len(test_cosmos),
            "cosmos_format": True,
            "video_specs": {
                "fps": 4,
                "max_duration": 30,
                "resolution": "720x480"
            }
        }
        
        metadata_file = self.dataset_dir / "dataset_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  ✅ Saved: train.jsonl ({len(train_cosmos)} samples)")
        print(f"  ✅ Saved: validation.jsonl ({len(val_cosmos)} samples)")
        print(f"  ✅ Saved: test.jsonl ({len(test_cosmos)} samples)")
        print(f"  ✅ Saved: dataset_metadata.json")
    
    def save_jsonl(self, data: List[Dict], filepath: Path):
        """Save data to JSONL file."""
        with open(filepath, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
    
    def create_dataset_summary(self):
        """Create a summary of the dataset."""
        print("📊 Creating dataset summary...")
        
        # Count samples in each split
        train_count = sum(1 for _ in open(self.dataset_dir / "train.jsonl"))
        val_count = sum(1 for _ in open(self.dataset_dir / "validation.jsonl"))
        test_count = sum(1 for _ in open(self.dataset_dir / "test.jsonl"))
        
        # Count classes
        classes = set()
        for filepath in [self.dataset_dir / "train.jsonl", self.dataset_dir / "validation.jsonl", self.dataset_dir / "test.jsonl"]:
            with open(filepath, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    classes.add(data["class"])
        
        summary = f"""
# 🏆 Cosmos Football Video Dataset Summary

## 📊 Dataset Statistics
- **Total Samples**: {train_count + val_count + test_count}
- **Training**: {train_count} samples ({train_count/(train_count + val_count + test_count)*100:.1f}%)
- **Validation**: {val_count} samples ({val_count/(train_count + val_count + test_count)*100:.1f}%)
- **Test**: {test_count} samples ({test_count/(train_count + val_count + test_count)*100:.1f}%)

## 🏷️ Classes ({len(classes)})
{chr(10).join(f"- {cls}" for cls in sorted(classes))}

## 📁 Files
- `train.jsonl` - Training data
- `validation.jsonl` - Validation data  
- `test.jsonl` - Test data
- `dataset_metadata.json` - Dataset metadata

## 🎯 Cosmos Format
- **Video FPS**: 4 (required by Cosmos-Reason1-7B)
- **Max Duration**: 30 seconds
- **Resolution**: 720x480
- **Format**: JSONL with video paths and annotations

## 🚀 Next Steps
1. Review the dataset splits
2. Start training with `05_training/fine_tune.py`
3. Evaluate with `06_evaluation/evaluate.py`
"""
        
        summary_file = self.dataset_dir / "README.md"
        with open(summary_file, 'w') as f:
            f.write(summary)
        
        print(f"  ✅ Created: {summary_file}")
    
    def build_dataset(self):
        """Build the complete dataset."""
        print("🏗️  Building Football Video Dataset")
        print("=" * 50)
        
        # Load annotations
        annotations = self.load_annotations()
        if not annotations:
            print("❌ No annotations found. Please run the annotation tool first.")
            return
        
        # Find video files
        video_files = self.find_video_files()
        
        # Create splits
        train_data, val_data, test_data = self.create_dataset_splits(annotations, video_files)
        
        # Save dataset
        self.save_dataset(train_data, val_data, test_data)
        
        # Create summary
        self.create_dataset_summary()
        
        print("=" * 50)
        print("✅ Dataset building completed!")
        print("")
        print("📁 Dataset files created in: 04_dataset/")
        print("🚀 Ready for training!")

def main():
    parser = argparse.ArgumentParser(description="Build Football Video Dataset")
    parser.add_argument("--project-root", default="/Users/Genesis/Desktop/upwork/Nvidia-AI/project-cosmos-football", 
                       help="Project root directory")
    parser.add_argument("--train-split", type=float, default=0.7, 
                       help="Training split ratio")
    parser.add_argument("--val-split", type=float, default=0.15, 
                       help="Validation split ratio")
    parser.add_argument("--test-split", type=float, default=0.15, 
                       help="Test split ratio")
    
    args = parser.parse_args()
    
    # Validate splits
    if abs(args.train_split + args.val_split + args.test_split - 1.0) > 1e-6:
        print("❌ Error: Train + Val + Test splits must equal 1.0")
        return
    
    builder = FootballDatasetBuilder(args.project_root)
    builder.train_split = args.train_split
    builder.val_split = args.val_split
    builder.test_split = args.test_split
    
    builder.build_dataset()

if __name__ == "__main__":
    main()