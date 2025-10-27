#!/usr/bin/env python3
"""
Prepare Football Dataset for Cosmos RL Training
Converts SFT format to LLaVA format required by Cosmos RL
"""

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def convert_to_llava(input_jsonl: str, output_json: str):
    """Convert SFT JSONL to LLaVA JSON format."""
    
    llava_data = []
    
    with open(input_jsonl, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            
            sft_item = json.loads(line)
            
            # Convert to LLaVA format
            llava_item = {
                "id": Path(sft_item["video"]).stem,
                "image": None,
                "video": sft_item["video"],
                "conversations": [
                    {
                        "from": "human",
                        "value": sft_item["prompt"]
                    },
                    {
                        "from": "gpt",
                        "value": sft_item["completion"]
                    }
                ]
            }
            
            llava_data.append(llava_item)
    
    # Save LLaVA format
    with open(output_json, 'w') as f:
        json.dump(llava_data, f, indent=2)
    
    print(f"✅ Converted {len(llava_data)} examples to LLaVA format")
    return len(llava_data)


def main():
    """Prepare training and validation datasets."""
    
    # Paths
    dataset_dir = project_root / "04_dataset"
    
    train_jsonl = dataset_dir / "train.jsonl"
    val_jsonl = dataset_dir / "validation.jsonl"
    
    train_llava = dataset_dir / "train_llava.json"
    val_llava = dataset_dir / "val_llava.json"
    
    # Convert training data
    if train_jsonl.exists():
        print("📊 Converting training data to LLaVA format...")
        count = convert_to_llava(train_jsonl, train_llava)
        print(f"✅ Created {train_llava} with {count} examples")
    else:
        print(f"⚠️  Training file not found: {train_jsonl}")
    
    # Convert validation data
    if val_jsonl.exists():
        print("📊 Converting validation data to LLaVA format...")
        count = convert_to_llava(val_jsonl, val_llava)
        print(f"✅ Created {val_llava} with {count} examples")
    else:
        print(f"⚠️  Validation file not found: {val_jsonl}")
    
    print("\n✅ Dataset preparation complete!")
    print(f"📁 LLaVA datasets ready in: {dataset_dir}")


if __name__ == "__main__":
    main()

