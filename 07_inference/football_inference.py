#!/usr/bin/env python3
"""
Football Video Analysis Inference Script
Uses fine-tuned LoRA weights for football video analysis
Based on Cosmos-Reason1-7B with LoRA adapters
"""

import os
import json
import torch
from pathlib import Path
from typing import Dict, Any, List, Optional
import argparse
from dotenv import load_dotenv
import time

# Cosmos-specific imports
from transformers import AutoProcessor
from vllm import LLM, SamplingParams, LoRARequest
from qwen_vl_utils import process_vision_info


class FootballVideoInference:
    """Football video analysis inference with LoRA weights."""
    
    def __init__(self, model_path: str, lora_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.lora_path = lora_path
        self.device = device
        self.llm = None
        self.processor = None
        self.lora_adapter_name = "football_analysis_lora"
        
        # Load model and LoRA weights
        self._load_model()
    
    def _load_model(self):
        """Load the base model and LoRA weights."""
        print(f"🚀 Loading Cosmos-Reason1-7B model from {self.model_path}")
        print(f"🔧 Loading LoRA weights from {self.lora_path}")
        
        # Load the base model with vLLM
        self.llm = LLM(
            model=self.model_path,
            limit_mm_per_prompt={"image": 10, "video": 10},
            dtype="bfloat16",  # BF16 as per Cosmos-Reason1-7B specs
            gpu_memory_utilization=0.8,
            enable_lora=True,  # Enable LoRA support
        )
        
        # Load the processor for chat template
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        
        print("✅ Base model loaded successfully!")
        print("✅ LoRA support enabled!")
    
    def create_football_prompt(self) -> str:
        """Create the football analysis prompt."""
        return """Analyze the following football clip. Identify all significant events including goals, cards, and shots. For each event, provide a detailed description including player, team, and jersey info, the event class, and precise start/end timestamps. Output *only* a valid JSON array."""
    
    def process_video(self, video_path: str) -> Dict[str, Any]:
        """Process a single football video and return analysis."""
        print(f"🎬 Processing video: {video_path}")
        
        # Create video messages with Cosmos format
        video_messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that can analyze football videos and identify significant events including goals, cards, and shots. For each event, provide a detailed description including player, team, and jersey info, the event class, and precise start/end timestamps. Output *only* a valid JSON array."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": self.create_football_prompt()
                    },
                    {
                        "type": "video",
                        "video": f"file://{os.path.abspath(video_path)}",
                        "fps": 4,  # Use fps=4 as per Cosmos-Reason1-7B specs
                    }
                ]
            },
        ]
        
        # Process the prompt
        prompt = self.processor.apply_chat_template(
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
            temperature=0.6,
            top_p=0.95,
            repetition_penalty=1.05,
            max_tokens=4096,  # Use 4096+ tokens as per Cosmos specs
        )
        
        # Prepare inputs for vLLM with LoRA
        llm_inputs = {
            "prompt": prompt,
            "multi_modal_data": mm_data,
            "mm_processor_kwargs": video_kwargs,
        }
        
        # Add LoRA adapter to the request
        lora_request = LoRARequest(
            lora_name=self.lora_adapter_name,
            lora_int_id=1,
            lora_local_path=self.lora_path
        )
        llm_inputs["lora_request"] = lora_request
        
        # Generate response
        start_time = time.time()
        outputs = self.llm.generate([llm_inputs], sampling_params=sampling_params)
        generation_time = time.time() - start_time
        
        generated_text = outputs[0].outputs[0].text
        
        # Parse the JSON response
        try:
            analysis_result = json.loads(generated_text)
        except json.JSONDecodeError:
            # If JSON parsing fails, return the raw text
            analysis_result = {
                "raw_output": generated_text,
                "parse_error": "Failed to parse JSON output"
            }
        
        result = {
            "video_path": video_path,
            "analysis": analysis_result,
            "generation_time": generation_time,
            "model_info": {
                "base_model": self.model_path,
                "lora_adapter": self.lora_path,
                "adapter_name": self.lora_adapter_name
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return result
    
    def process_batch(self, video_paths: List[str], output_dir: str) -> List[Dict[str, Any]]:
        """Process multiple videos and save results."""
        print(f"📁 Processing {len(video_paths)} videos")
        print(f"📁 Output directory: {output_dir}")
        
        results = []
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for i, video_path in enumerate(video_paths):
            print(f"\n📹 Processing {i+1}/{len(video_paths)}: {Path(video_path).name}")
            
            try:
                result = self.process_video(video_path)
                results.append(result)
                
                # Save individual result
                video_name = Path(video_path).stem
                result_file = output_path / f"{video_name}_analysis.json"
                with open(result_file, 'w') as f:
                    json.dump(result, f, indent=2)
                
                print(f"  ✅ Analysis saved to: {result_file}")
                
            except Exception as e:
                print(f"  ❌ Error processing {video_path}: {e}")
                error_result = {
                    "video_path": video_path,
                    "error": str(e),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                results.append(error_result)
        
        # Save batch results
        batch_file = output_path / "batch_analysis_results.json"
        with open(batch_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📊 Batch results saved to: {batch_file}")
        return results


def load_environment():
    """Load environment variables from .env file."""
    load_dotenv()
    
    config = {
        "model_path": os.getenv("MODEL_PATH", "nvidia/Cosmos-Reason1-7B"),
        "lora_path": os.getenv("LORA_PATH", "../05_training/checkpoints/football_sft"),
        "device": os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"),
        "temperature": float(os.getenv("TEMPERATURE", "0.6")),
        "top_p": float(os.getenv("TOP_P", "0.95")),
        "max_tokens": int(os.getenv("MAX_TOKENS", "4096"))
    }
    
    return config


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="Football Video Analysis Inference")
    parser.add_argument("--video_path", type=str, required=True,
                       help="Path to the input video file")
    parser.add_argument("--output_dir", type=str, default="./inference_results",
                       help="Directory to save results")
    parser.add_argument("--model_path", type=str, default="nvidia/Cosmos-Reason1-7B",
                       help="Path to base model")
    parser.add_argument("--lora_path", type=str, default="../05_training/checkpoints/football_sft",
                       help="Path to LoRA weights")
    parser.add_argument("--batch", action="store_true",
                       help="Process multiple videos from a directory")
    parser.add_argument("--video_dir", type=str,
                       help="Directory containing videos to process (for batch mode)")
    
    args = parser.parse_args()
    
    print("🏈 Football Video Analysis Inference")
    print("=" * 50)
    
    # Load configuration
    config = load_environment()
    
    # Override with command line arguments
    model_path = args.model_path or config["model_path"]
    lora_path = args.lora_path or config["lora_path"]
    
    print(f"📋 Model: {model_path}")
    print(f"📋 LoRA: {lora_path}")
    print(f"📋 Device: {config['device']}")
    
    # Create inference instance
    try:
        inference = FootballVideoInference(
            model_path=model_path,
            lora_path=lora_path,
            device=config["device"]
        )
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return 1
    
    # Process video(s)
    if args.batch and args.video_dir:
        # Batch processing
        video_dir = Path(args.video_dir)
        video_paths = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi")) + list(video_dir.glob("*.mov"))
        
        if not video_paths:
            print(f"❌ No videos found in {video_dir}")
            return 1
        
        results = inference.process_batch([str(p) for p in video_paths], args.output_dir)
        
        print(f"\n✅ Batch processing completed!")
        print(f"📊 Processed {len(results)} videos")
        
    else:
        # Single video processing
        if not os.path.exists(args.video_path):
            print(f"❌ Video file not found: {args.video_path}")
            return 1
        
        result = inference.process_video(args.video_path)
        
        # Save result
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        video_name = Path(args.video_path).stem
        result_file = output_path / f"{video_name}_analysis.json"
        
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\n✅ Analysis completed!")
        print(f"📄 Result saved to: {result_file}")
        
        # Print the JSON output
        print("\n" + "=" * 50)
        print("📊 FOOTBALL VIDEO ANALYSIS RESULT")
        print("=" * 50)
        print(json.dumps(result["analysis"], indent=2))
    
    return 0


if __name__ == "__main__":
    exit(main())
