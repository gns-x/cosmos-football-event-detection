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
        # Create the football analysis prompt for real video analysis
        football_prompt = """Analyze this football video and identify significant events by watching the actual video content.

VALID EVENT CLASSES:
- goal: When a player scores a goal
- penalty_shot: Penalty kick attempts
- red_card: Red card shown to a player
- yellow_card: Yellow card shown to a player  
- shot_on_target: Shots that are on target (saved or hit post)
- goal_line_event: Goal line technology moments, close calls
- hat_trick: When a player scores 3 goals in one match
- woodworks: Shots that hit the post or crossbar

CRITICAL REQUIREMENTS:
1. Watch the video carefully and identify events based on what you actually see
2. Provide accurate timestamps based on when events occur in the video
3. Only identify events that are clearly visible in the video
4. If no clear events are visible, return an empty array []
5. Do not guess or make assumptions - only report what you observe

For each event you observe, provide:
- event: One of the valid classes above
- description: Brief description of what you actually see happening
- start_time: Exact timestamp when event starts (format: "MM:SS")
- end_time: Exact timestamp when event ends (format: "MM:SS")

Output ONLY a valid JSON array with no additional text."""
        
        # Create conversation format
        messages = [
            {
                "role": "system",
                "content": "You are a professional football video analyst. You must analyze video content accurately and provide precise timestamps based on what you observe. Only report events that are clearly visible in the video. Do not make assumptions or provide placeholder data."
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
        valid_event_classes = {
            "goal", "penalty_shot", "red_card", "yellow_card", 
            "shot_on_target", "goal_line_event", "hat_trick", "woodworks"
        }
        
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\[.*\]', assistant_response, re.DOTALL)
            if json_match:
                events = json.loads(json_match.group())
                
                # Validate events but don't modify them - keep original analysis
                validated_events = []
                for event in events:
                    if isinstance(event, dict):
                        # Validate event class
                        event_type = event.get("event", event.get("type", "")).lower()
                        if event_type in valid_event_classes:
                            validated_event = {
                                "event": event_type,
                                "description": event.get("description", ""),
                                "start_time": event.get("start_time", ""),
                                "end_time": event.get("end_time", "")
                            }
                            validated_events.append(validated_event)
                        else:
                            print(f"  ⚠️  Invalid event class: {event_type}")
                
                events = validated_events if validated_events else []
            else:
                # No JSON found - model didn't provide structured output
                print(f"  ⚠️  No JSON structure found in response")
                events = []
        except json.JSONDecodeError as e:
            # JSON parsing failed - model output is malformed
            print(f"  ⚠️  JSON parsing failed: {e}")
            events = []
        
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