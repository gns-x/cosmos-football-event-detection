#!/usr/bin/env python3
"""
Inference script for running the fine-tuned Cosmos model on new videos.
"""

import os
import json
import torch
from pathlib import Path
from typing import Dict, Any, List
import argparse
from dotenv import load_dotenv

# Cosmos-specific imports
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

def load_environment():
    """Load environment variables from .env file."""
    load_dotenv()
    
    # Get configuration from environment variables
    config = {
        "model_path": os.getenv("MODEL_PATH", "../05_training/checkpoints"),
        "device": os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"),
        "max_length": int(os.getenv("MAX_LENGTH", "2048")),
        "temperature": float(os.getenv("TEMPERATURE", "0.7")),
        "top_p": float(os.getenv("TOP_P", "0.9"))
    }
    
    return config

def load_model(model_path: str, device: str):
    """Load the Cosmos model using vLLM."""
    print(f"Loading Cosmos model from {model_path}...")
    
    # Load the Cosmos model with vLLM
    llm = LLM(
        model=model_path,
        limit_mm_per_prompt={"image": 10, "video": 10},
        dtype="bfloat16",  # BF16 as per Cosmos-Reason1-7B specs
        gpu_memory_utilization=0.8,
    )
    
    # Load the processor for chat template
    processor = AutoProcessor.from_pretrained(model_path)
    
    print("Cosmos model loaded successfully!")
    return llm, processor

def process_video(video_path: str, model_data, config: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single video using Cosmos model."""
    
    print(f"Processing video: {video_path}")
    
    llm, processor = model_data
    
    # Create video messages with Cosmos format
    video_messages = [
        {
            "role": "system", 
            "content": "You are a helpful assistant. Answer the question in the following format: \n\n."
        },
        {
            "role": "user", 
            "content": [
                {"type": "text", "text": "Describe the football action in this video."},
                {
                    "type": "video", 
                    "video": f"file://{os.path.abspath(video_path)}",
                    "fps": 4,  # Use fps=4 as per Cosmos-Reason1-7B specs
                }
            ]
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
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs
    
    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=config.get("temperature", 0.6),
        top_p=config.get("top_p", 0.95),
        repetition_penalty=1.05,
        max_tokens=4096,  # Use 4096+ tokens as per Cosmos specs
    )
    
    # Prepare inputs for vLLM
    llm_inputs = {
        "prompt": prompt,
        "multi_modal_data": mm_data,
        "mm_processor_kwargs": video_kwargs,
    }
    
    # Generate response
    outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
    generated_text = outputs[0].outputs[0].text
    
    result = {
        "video_path": video_path,
        "description": generated_text,
        "confidence": 0.85,  # Placeholder confidence
        "processing_time": 2.5  # Placeholder time
    }
    
    return result

def run_inference_on_video(video_path: str, model, config: Dict[str, Any]) -> Dict[str, Any]:
    """Run inference on a single video."""
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    result = process_video(video_path, model, config)
    return result

def run_inference_on_directory(directory_path: str, model, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Run inference on all videos in a directory."""
    
    results = []
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    
    directory = Path(directory_path)
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(directory.glob(f"*{ext}"))
    
    print(f"Found {len(video_files)} video files")
    
    for video_file in video_files:
        try:
            result = run_inference_on_video(str(video_file), model, config)
            results.append(result)
        except Exception as e:
            print(f"Error processing {video_file}: {e}")
            results.append({
                "video_path": str(video_file),
                "error": str(e),
                "description": None
            })
    
    return results

def save_results(results: List[Dict[str, Any]], output_file: str):
    """Save inference results to JSON file."""
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")

def main():
    """Main inference function."""
    
    parser = argparse.ArgumentParser(description="Run Cosmos inference on football videos")
    parser.add_argument("--input", required=True, help="Path to video file or directory")
    parser.add_argument("--output", default="inference_results.json", help="Output file for results")
    parser.add_argument("--batch", action="store_true", help="Process directory of videos")
    
    args = parser.parse_args()
    
    # Load environment configuration
    config = load_environment()
    print(f"Configuration: {config}")
    
    # Load model
    model = load_model(config["model_path"], config["device"])
    
    # Run inference
    if args.batch or os.path.isdir(args.input):
        print("Running batch inference on directory...")
        results = run_inference_on_directory(args.input, model, config)
    else:
        print("Running inference on single video...")
        result = run_inference_on_video(args.input, model, config)
        results = [result]
    
    # Save results
    save_results(results, args.output)
    
    # Print summary
    successful = len([r for r in results if "error" not in r])
    print(f"\nInference completed!")
    print(f"Successfully processed: {successful}/{len(results)} videos")
    
    if successful > 0:
        avg_confidence = sum(r.get("confidence", 0) for r in results if "confidence" in r) / successful
        print(f"Average confidence: {avg_confidence:.3f}")

if __name__ == "__main__":
    main()
