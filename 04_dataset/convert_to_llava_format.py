#!/usr/bin/env python3
"""
Convert SFT Dataset to LLaVA Format for Cosmos RL
Converts our (prompt, completion) format to LLaVA's conversations format
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any


def convert_sft_to_llava(input_file: str, output_file: str):
    """Convert SFT format to LLaVA conversations format."""
    
    llava_data = []
    
    with open(input_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue
                
            # Parse SFT format
            sft_item = json.loads(line)
            
            # Convert to LLaVA format
            llava_item = {
                "id": Path(sft_item["video"]).stem,
                "image": None,  # No images
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
    with open(output_file, 'w') as f:
        json.dump(llava_data, f, indent=2)
    
    print(f"✅ Converted {len(llava_data)} examples to LLaVA format")
    print(f"📁 Saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Convert SFT to LLaVA format")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file")
    
    args = parser.parse_args()
    
    convert_sft_to_llava(args.input, args.output)


if __name__ == "__main__":
    main()

