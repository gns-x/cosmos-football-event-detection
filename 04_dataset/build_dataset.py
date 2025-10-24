#!/usr/bin/env python3
"""
Script to create train/val/test splits from processed videos and annotations.
"""

import json
import os
import random
from pathlib import Path
from typing import List, Dict, Any

def load_annotations(annotation_dir: str) -> List[Dict[str, Any]]:
    """Load all annotation JSON files from the ground_truth_json directory."""
    annotations = []
    annotation_path = Path(annotation_dir)
    
    for json_file in annotation_path.glob("*.json"):
        with open(json_file, 'r') as f:
            data = json.load(f)
            annotations.append(data)
    
    return annotations

def create_dataset_splits(annotations: List[Dict[str, Any]], 
                         train_ratio: float = 0.7, 
                         val_ratio: float = 0.15, 
                         test_ratio: float = 0.15) -> Dict[str, List[Dict[str, Any]]]:
    """Split annotations into train/val/test sets."""
    
    # Shuffle annotations
    random.shuffle(annotations)
    
    total_count = len(annotations)
    train_count = int(total_count * train_ratio)
    val_count = int(total_count * val_ratio)
    
    splits = {
        'train': annotations[:train_count],
        'validation': annotations[train_count:train_count + val_count],
        'test': annotations[train_count + val_count:]
    }
    
    return splits

def save_jsonl(data: List[Dict[str, Any]], filepath: str):
    """Save data as JSONL format."""
    with open(filepath, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def main():
    """Main function to build the dataset."""
    
    # Configuration
    annotation_dir = "../03_annotation/ground_truth_json"
    output_dir = "."
    
    print("Loading annotations...")
    annotations = load_annotations(annotation_dir)
    print(f"Loaded {len(annotations)} annotation files")
    
    print("Creating dataset splits...")
    splits = create_dataset_splits(annotations)
    
    print(f"Train set: {len(splits['train'])} samples")
    print(f"Validation set: {len(splits['validation'])} samples")
    print(f"Test set: {len(splits['test'])} samples")
    
    print("Saving dataset files...")
    save_jsonl(splits['train'], os.path.join(output_dir, "train.jsonl"))
    save_jsonl(splits['validation'], os.path.join(output_dir, "validation.jsonl"))
    save_jsonl(splits['test'], os.path.join(output_dir, "test.jsonl"))
    
    print("Dataset creation completed!")

if __name__ == "__main__":
    main()
