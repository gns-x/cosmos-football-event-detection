#!/usr/bin/env python3
"""
Professional Football Video Analysis Inference with LoRA
Processes all videos from data collection folder
Based on your exact specifications
"""

import os
import sys
import json
import argparse
import torch
import transformers
from pathlib import Path
from transformers import AutoTokenizer, AutoProcessor
from peft import PeftModel


def find_all_videos(data_collection_dir: str):
    """Find all videos in the data collection directory."""
    videos = []
    data_dir = Path(data_collection_dir)
    
    if not data_dir.exists():
        return videos
    
    # Search recursively for video files
    for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
        videos.extend(data_dir.rglob(ext))
    
    return sorted(videos)


def load_model_with_lora(model_path: str, lora_path: str = None):
    """Load base model and LoRA adapter."""
    print(f"🚀 Loading Cosmos-Reason1-7B base model with LoRA adapter...")
    
    # Load tokenizer and processor
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        processor = AutoProcessor.from_pretrained(
            model_path, 
            trust_remote_code=True
        )
        print("✅ Tokenizer and processor loaded")
    except Exception as e:
        print(f"❌ Failed to load tokenizer/processor: {e}")
        return None, None, None
    
    # Load base model
    try:
        print("📋 Loading base Cosmos-Reason1-7B model...")
        model = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        print("✅ Base model loaded")
    except Exception as e:
        print(f"❌ Failed to load base model: {e}")
        return None, None, None
    
    # Load LoRA adapter if available
    if lora_path and Path(lora_path).exists() and (Path(lora_path) / "adapter_config.json").exists():
        print(f"📥 Loading LoRA adapter from: {lora_path}")
        try:
            model = PeftModel.from_pretrained(model, str(lora_path))
            print("✅ LoRA adapter loaded and merged")
        except Exception as e:
            print(f"❌ Failed to load LoRA adapter: {e}")
            print("⚠️  Continuing with base model only")
    else:
        print("⚠️  No LoRA adapter found, using base model only")
    
    # Set model to evaluation mode
    model.eval()
    print("✅ Model ready for inference")
    
    return model, tokenizer, processor


def process_video(video_path: str, model, tokenizer, processor):
    """Process a single video and return analysis."""
    print(f"\n🎬 Processing: {Path(video_path).name}")
    
    try:
        # Create the football analysis prompt
        football_prompt = "Analyze this football video clip and identify all significant events including goals, penalty shots, cards, and shots. For each event, provide: event type, description, start time, and end time. Output as JSON array."
        
        # Create conversation format
        messages = [
            {
                "role": "system",
                "content": "You are a football video analysis expert. Analyze the video and identify significant events."
            },
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": football_prompt},
                    {"type": "video", "video": str(video_path)}
                ]
            }
        ]
        
        # Process with the model
        text = processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Process video and text
        inputs = processor(
            text=[text],
            videos=[str(video_path)],
            padding=True,
            return_tensors="pt"
        )
        
        # Move to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant response
        if "Assistant:" in response:
            assistant_response = response.split("Assistant:")[-1].strip()
        else:
            assistant_response = response
        
        print(f"  ✅ Generated response: {assistant_response[:100]}...")
        
        # Parse response as JSON if possible
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\[.*\]', assistant_response, re.DOTALL)
            if json_match:
                events = json.loads(json_match.group())
            else:
                # Fallback: create a single event from the response
                events = [{
                    "event": "analysis",
                    "description": assistant_response,
                    "start_time": "0:00",
                    "end_time": "0:00"
                }]
        except json.JSONDecodeError:
            # Fallback: create a single event from the response
            events = [{
                "event": "analysis", 
                "description": assistant_response,
                "start_time": "0:00",
                "end_time": "0:00"
            }]
        
        return {
            "video_file": Path(video_path).name,
            "video_path": str(video_path),
            "events": events,
            "raw_response": assistant_response,
            "model_version": "cosmos-reason1-7b-lora"
        }
        
    except Exception as e:
        print(f"  ❌ Error processing video: {e}")
        return {
            "video_file": Path(video_path).name,
            "video_path": str(video_path),
            "error": str(e),
            "events": []
        }


def main():
    """Main inference function with batch processing."""
    parser = argparse.ArgumentParser(description="Football Video Analysis Inference")
    parser.add_argument("--video_path", type=str, help="Path to single video")
    parser.add_argument("--data_collection_dir", type=str, 
                       default="../01_data_collection/raw_videos",
                       help="Directory containing videos")
    parser.add_argument("--output_dir", type=str, default="./inference_results",
                       help="Output directory")
    parser.add_argument("--model_path", type=str, default="nvidia/Cosmos-Reason1-7B",
                       help="Model path")
    parser.add_argument("--lora_path", type=str, default="../05_training/checkpoints/football_sft",
                       help="LoRA adapter path")
    parser.add_argument("--process_all", action="store_true",
                       help="Process all videos in data collection directory")
    
    args = parser.parse_args()
    
    print("🏈 Professional Football Video Analysis Inference")
    print("=" * 60)
    print(f"📋 Model: {args.model_path}")
    print(f"📁 LoRA: {args.lora_path}")
    print(f"📊 Output: {args.output_dir}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model, tokenizer, processor = load_model_with_lora(args.model_path, args.lora_path)
    
    if model is None:
        print("❌ Failed to load model. Exiting.")
        return 1
    
    # Determine videos to process
    if args.process_all:
        videos = find_all_videos(args.data_collection_dir)
        print(f"📊 Found {len(videos)} videos to process")
    elif args.video_path:
        videos = [Path(args.video_path)]
    else:
        print("❌ No video specified. Use --video_path or --process_all")
        return 1
    
    # Process videos
    results = []
    for i, video_path in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}] Processing video...")
        
        result = process_video(str(video_path), model, tokenizer, processor)
        results.append(result)
        
        # Save individual result
        video_name = Path(video_path).stem
        individual_file = output_dir / f"{video_name}_analysis.json"
        with open(individual_file, 'w') as f:
            json.dump(result, f, indent=2)
    
    # Save combined results
    combined_file = output_dir / "all_analyses.json"
    with open(combined_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Inference completed!")
    print(f"📁 Results saved to: {output_dir}")
    print(f"📊 Processed {len(results)} videos")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())