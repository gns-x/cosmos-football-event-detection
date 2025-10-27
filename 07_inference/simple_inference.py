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
from pathlib import Path
from vllm import LLM, SamplingParams, LoRARequest
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info


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


def process_video(video_path: str, llm, processor, lora_path: str = None):
    """Process a single video and return analysis."""
    print(f"\n🎬 Processing: {Path(video_path).name}")
    
    # Create the football analysis prompt
    football_prompt = "Analyze the following football clip. Identify all significant events including goals, cards, and shots. For each event, provide a detailed description including player, team, and jersey info, the event class, and precise start/end timestamps. Output *only* a valid JSON array."
    
    # Create video messages
    video_messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that can analyze football videos and identify significant events including goals, cards, and shots. For each event, provide a detailed description including player, team, and jersey info, the event class, and precise start/end timestamps. Output *only* a valid JSON array."
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": football_prompt},
                {
                    "type": "video",
                    "video": f"file://{os.path.abspath(video_path)}",
                    "fps": 4,
                }
            ],
        },
    ]
    
    # Process the prompt
    prompt = processor.apply_chat_template(
        video_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    # Process vision information
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        video_messages,
        return_video_kwargs=True
    )
    
    # Prepare multimodal data
    mm_data = {}
    if video_inputs is not None:
        mm_data["video"] = video_inputs
    
    # Set up sampling parameters
    sampling_params = SamplingParams(
        max_tokens=4096,
        temperature=0.6,
        top_p=0.95,
    )
    
    # Prepare inputs for vLLM
    llm_inputs = {
        "prompt": prompt,
        "multi_modal_data": mm_data,
        "mm_processor_kwargs": video_kwargs,
    }
    
    # Add LoRA adapter if available
    if lora_path and os.path.exists(lora_path):
        lora_request = LoRARequest(
            lora_name="football_analysis_lora",
            lora_int_id=1,
            lora_local_path=lora_path
        )
        llm_inputs["lora_request"] = lora_request
    
    # Generate response
    outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
    generated_json_string = outputs[0].outputs[0].text
    
    # Parse JSON
    try:
        analysis = json.loads(generated_json_string)
    except json.JSONDecodeError:
        analysis = {"raw_output": generated_json_string}
    
    return {
        "video_path": video_path,
        "video_name": Path(video_path).name,
        "analysis": analysis
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
                       help="Process all videos")
    
    args = parser.parse_args()
    
    print("🏈 Football Video Analysis - Professional Inference")
    print("=" * 60)
    print(f"📋 Base Model: {args.model_path}")
    print(f"📋 LoRA Path: {args.lora_path}")
    
    # Determine videos to process
    if args.process_all or not args.video_path:
        videos = find_all_videos(args.data_collection_dir)
        if not videos:
            print(f"❌ No videos found in {args.data_collection_dir}")
            return 1
        print(f"📁 Found {len(videos)} videos to process")
    else:
        if not os.path.exists(args.video_path):
            print(f"❌ Video not found: {args.video_path}")
            return 1
        videos = [args.video_path]
    
    # Load the base model with LoRA support
    print("\n🚀 Loading model with LoRA support...")
    llm = LLM(
        model=args.model_path,
        limit_mm_per_prompt={"video": 10},
        enable_lora=True,
        dtype="bfloat16",
        gpu_memory_utilization=0.8,
    )
    
    # Load processor
    processor = AutoProcessor.from_pretrained(args.model_path)
    
    print("✅ Model loaded successfully!")
    
    # Process all videos
    results = []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, video in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}] Processing video...")
        try:
            result = process_video(video, llm, processor, args.lora_path)
            results.append(result)
            
            # Save individual result
            video_name = Path(video).stem
            result_file = output_dir / f"{video_name}_analysis.json"
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"✅ Saved: {result_file}")
            
        except Exception as e:
            print(f"❌ Error processing {video}: {e}")
            results.append({
                "video_path": str(video),
                "error": str(e)
            })
    
    # Save batch summary
    batch_summary = {
        "total_videos": len(videos),
        "processed": len([r for r in results if "error" not in r]),
        "errors": len([r for r in results if "error" in r]),
        "results": results
    }
    
    batch_file = output_dir / "batch_summary.json"
    with open(batch_file, 'w') as f:
        json.dump(batch_summary, f, indent=2)
    
    print(f"\n📊 Batch Summary: {batch_file}")
    print(f"✅ Processed {batch_summary['processed']}/{batch_summary['total_videos']} videos")


if __name__ == "__main__":
    sys.exit(main())
