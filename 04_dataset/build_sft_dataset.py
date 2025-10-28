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
import re

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

    def _normalize(self, name: str) -> str:
        name = name.lower()
        name = re.sub(r"[_\-]+", " ", name)
        name = re.sub(r"[^a-z0-9 ]+", "", name)
        name = re.sub(r"\s+", " ", name).strip()
        return name

    def _resolve_video_path(self, video_stem: str, event_class: str) -> Tuple[Path, str]:
        """Find the best matching video file in processed first, else raw. Returns (path, class)."""
        norm_target = self._normalize(video_stem.replace("_processed", ""))
        # 1) Search processed
        proc_class_dir = self.processed_videos_dir / event_class
        if proc_class_dir.exists():
            candidates = list(proc_class_dir.glob("*.mp4"))
            for cand in candidates:
                stem = cand.stem.replace("_processed", "")
                if self._normalize(stem) == norm_target:
                    return cand, event_class
            # fallback: substring match
            for cand in candidates:
                stem = cand.stem.replace("_processed", "")
                n = self._normalize(stem)
                if norm_target in n or n in norm_target:
                    return cand, event_class
        # 2) Search raw
        raw_class_dir = self.project_root / "01_data_collection" / "raw_videos" / event_class
        if raw_class_dir.exists():
            candidates = list(raw_class_dir.glob("*.mp4"))
            for cand in candidates:
                if self._normalize(cand.stem) == norm_target:
                    return cand, event_class
            for cand in candidates:
                n = self._normalize(cand.stem)
                if norm_target in n or n in norm_target:
                    return cand, event_class
        return None, event_class
    
    def load_annotations(self) -> List[Dict[str, Any]]:
        """Load and normalize all annotation files to {'video': str, 'annotations': list}."""
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
                    
                    # Normalize both formats into a common structure
                    if isinstance(data, list):
                        # Direct array of annotations; infer video name from filename
                        stem = annotation_file.stem
                        # Strip trailing _annotations if present
                        if stem.endswith("_annotations"):
                            stem = stem[:-12]
                        annotations.append({
                            "video": stem,
                            "annotations": data
                        })
                    elif isinstance(data, dict):
                        ann_list = data.get("annotations") or []
                        video_name = data.get("video")
                        if not video_name:
                            # Fallback to filename
                            stem = annotation_file.stem
                            if stem.endswith("_annotations"):
                                stem = stem[:-12]
                            video_name = stem
                        annotations.append({
                            "video": video_name,
                            "annotations": ann_list
                        })
                    else:
                        print(f"  ⚠️  Unknown format in {annotation_file.name}")
                        continue
                        
                print(f"  ✅ Loaded: {annotation_file.name}")
            except Exception as e:
                print(f"  ❌ Error loading {annotation_file.name}: {e}")
        
        print(f"📊 Total annotations loaded: {len(annotations)}")
        return annotations
    
    def find_video_files(self) -> Dict[str, List[Path]]:
        """Find all video files organized by class (processed or raw)."""
        print("🔍 Finding video files...")
        
        video_files = {}
        video_count = 0
        
        # First try processed videos
        for video_file in self.processed_videos_dir.rglob("*.mp4"):
            class_name = video_file.parent.name
            if class_name not in video_files:
                video_files[class_name] = []
            video_files[class_name].append(video_file)
            video_count += 1
        
        # If no processed videos, try raw videos
        if video_count == 0:
            print("  ⚠️  No processed videos found, checking raw videos...")
            raw_videos_dir = self.project_root / "01_data_collection" / "raw_videos"
            for video_file in raw_videos_dir.rglob("*.mp4"):
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
        """Create train/validation splits for SFT dataset.
        New strategy: iterate annotations and resolve the corresponding video path in processed/raw.
        """
        print("📊 Creating SFT dataset splits...")
        
        all_video_items = []
        for item in annotations:
            video_name = (item.get('video') or '').strip()
            ann_list = item.get('annotations', [])
            if not video_name or not ann_list:
                continue
            # Try to infer class from first annotation if not present
            class_name = None
            # Some saved files include event_class at top-level; try that first
            class_name = item.get('event_class')
            if not class_name and len(ann_list) > 0:
                ev = (ann_list[0].get('event') or '').strip()
                if ev:
                    class_name = ev
            # Fallback: try to guess from file prefix convention like "goal_..."
            if not class_name and '_' in video_name:
                possible = video_name.split('_', 1)[0]
                if (self.processed_videos_dir / possible).exists():
                    class_name = possible
            if not class_name:
                # Default unknown; skip gracefully
                print(f"  ⚠️  Could not infer class for {video_name}; skipping")
                continue
            
            video_path, resolved_class = self._resolve_video_path(video_name, class_name)
            if not video_path:
                print(f"  ⚠️  Skipping {video_name} - video file not found in processed/raw")
                continue
            
            all_video_items.append({
                'video_path': video_path,
                'annotations': ann_list
            })
        
        # Shuffle all videos
        random.shuffle(all_video_items)
        
        # Split into train/val
        total = len(all_video_items)
        train_size = int(total * self.train_split)
        
        train_data = all_video_items[:train_size]
        val_data = all_video_items[train_size:]
        
        print(f"  Total videos: {total}")
        print(f"    Train: {len(train_data)}, Val: {len(val_data)}")
        
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
        
        total_examples = train_count + val_count
        train_pct = (train_count/total_examples*100) if total_examples > 0 else 0.0
        val_pct = (val_count/total_examples*100) if total_examples > 0 else 0.0
        
        summary = f"""
# 🏆 Cosmos Football Video SFT Dataset Summary

## 📊 Dataset Statistics
- **Total Examples**: {total_examples}
- **Training**: {train_count} examples ({train_pct:.1f}%)
- **Validation**: {val_count} examples ({val_pct:.1f}%)

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
    parser.add_argument("--project-root", default=".", 
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
