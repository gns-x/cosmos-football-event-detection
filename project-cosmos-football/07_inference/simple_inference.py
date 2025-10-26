#!/usr/bin/env python3
"""
Simplified Football Video Analysis Inference
Demonstrates LoRA integration with vLLM
Based on your exact specifications
"""

import os
import json
from pathlib import Path
from vllm import LLM, SamplingParams, LoRARequest
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info


def main():
    """Simplified inference with LoRA weights."""
    print("🏈 Football Video Analysis - Simplified Inference")
    print("=" * 60)
    
    # Configuration
    MODEL_PATH = "nvidia/Cosmos-Reason1-7B"
    LORA_PATH = "../05_training/checkpoints/football_sft"
    VIDEO_PATH = "path/to/your/football_video.mp4"  # Replace with actual video path
    
    print(f"📋 Base Model: {MODEL_PATH}")
    print(f"📋 LoRA Path: {LORA_PATH}")
    print(f"📋 Video: {VIDEO_PATH}")
    
    # Load the base model with LoRA support
    print("\n🚀 Loading model with LoRA support...")
    llm = LLM(
        model=MODEL_PATH,
        limit_mm_per_prompt={"video": 10},
        enable_lora=True,  # Enable LoRA support
        dtype="bfloat16",
        gpu_memory_utilization=0.8,
    )
    
    # Load processor
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    
    print("✅ Model loaded successfully!")
    
    # Create the football analysis prompt (from Phase 3)
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
                    "video": f"file://{os.path.abspath(VIDEO_PATH)}",
                    "fps": 4,  # 4 FPS as required by Cosmos-Reason1-7B
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
        max_tokens=4096,  # Per model card
        temperature=0.6,
        top_p=0.95,
    )
    
    # Prepare inputs for vLLM
    llm_inputs = {
        "prompt": prompt,
        "multi_modal_data": mm_data,
        "mm_processor_kwargs": video_kwargs,
    }
    
    # *** THIS IS THE KEY ***
    # Add the LoRA adapter to the request
    lora_request = LoRARequest(
        lora_name="football_analysis_lora",
        lora_int_id=1,
        lora_local_path=LORA_PATH
    )
    llm_inputs["lora_request"] = lora_request
    
    print("\n🎬 Running inference with LoRA weights...")
    
    # Generate response
    outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
    generated_json_string = outputs[0].outputs[0].text
    
    print("\n" + "=" * 60)
    print("📊 FOOTBALL VIDEO ANALYSIS RESULT")
    print("=" * 60)
    print(generated_json_string)
    print("=" * 60)
    
    # This string *is* your final JSON output
    print("\n✅ Inference completed successfully!")
    print("📄 The generated JSON string above is your final output")
    
    # Try to parse and pretty-print the JSON
    try:
        parsed_result = json.loads(generated_json_string)
        print("\n📋 Parsed JSON Result:")
        print(json.dumps(parsed_result, indent=2))
    except json.JSONDecodeError:
        print("\n⚠️  Generated text is not valid JSON, but this is the raw output from the model")


if __name__ == "__main__":
    main()
