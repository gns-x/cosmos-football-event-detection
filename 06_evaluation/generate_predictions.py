#!/usr/bin/env python3
"""
Generate Predictions Script for Football Video Analysis
Uses the actual fine-tuned model for inference on test set
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
import sys
import time

# Import required packages
import os
# Set environment variable BEFORE importing to avoid config conflicts
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

try:
    # Import transformers first to avoid conflicts
    from transformers import AutoProcessor
    AutoProcessor
    
    # Now import vLLM
    from vllm import LLM, SamplingParams
    import vllm  # Import vllm module to check version
    from qwen_vl_utils import process_vision_info
    HAS_VLLM = True
    
    # Check if LoRA is available in this version
    if hasattr(vllm, 'lora'):
        from vllm.lora.request import LoRARequest
        HAS_LORA = True
    elif hasattr(vllm, 'LoRARequest'):
        from vllm import LoRARequest
        HAS_LORA = True
    else:
        # Try alternative import for newer versions
        try:
            from vllm.lora import LoRARequest
            HAS_LORA = True
        except ImportError:
            HAS_LORA = False
            
    print(f"✅ vLLM {vllm.__version__} loaded")
    print(f"✅ LoRA support: {HAS_LORA}")
except ImportError as e:
    print(f"❌ Failed to import vLLM packages: {e}")
    sys.exit(1)


class PredictionGenerator:
    """Generate predictions using the actual fine-tuned model."""
    
    def __init__(self, test_file: str, model_path: str, output_dir: str, lora_path: str = None):
        self.test_file = Path(test_file)
        self.model_path = Path(model_path)
        self.lora_path = Path(lora_path) if lora_path else None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.llm = None
        self.processor = None
        
        # Load test data
        self.test_data = self._load_test_data()
        print(f"📊 Loaded {len(self.test_data)} test examples")
        
        # Load model
        self._load_model()
    
    def _load_test_data(self) -> List[Dict[str, Any]]:
        """Load test dataset."""
        test_data = []
        if not self.test_file.exists():
            print(f"⚠️  Test file not found: {self.test_file}")
            return test_data
            
        with open(self.test_file, 'r') as f:
            for line in f:
                if line.strip():
                    test_data.append(json.loads(line))
        return test_data
    
    def _load_model(self):
        """Load the base Cosmos model (no LoRA for now)."""
        print("\n🚀 Loading Cosmos-Reason1-7B base model...")
        
        # Always use base model from HuggingFace
        base_model = "nvidia/Cosmos-Reason1-7B"
        
        # Load processor
        try:
            self.processor = AutoProcessor.from_pretrained(
                base_model, 
                trust_remote_code=True,
                use_auth_token=True
            )
            print("✅ Processor loaded")
        except Exception as e:
            print(f"❌ Failed to load processor: {e}")
            self.processor = None
            return
        
        # Load LLM (base model only, no LoRA)
        # Check if LoRA path exists and is valid
        lora_exists = self.lora_path and self.lora_path.exists() and (self.lora_path / "adapter_config.json").exists()
        
        if lora_exists and HAS_LORA:
            print(f"📥 LoRA adapter found at: {self.lora_path}")
            print("⚠️  Cosmos RL LoRA support not yet implemented, using base model")
        
        # Always use base model for now
        print("📋 Using base Cosmos-Reason1-7B model")
        try:
            # Try loading with trust_remote_code first
            try:
                self.llm = LLM(
                    model=base_model,
                    limit_mm_per_prompt={"video": 10},
                    dtype="bfloat16",
                    gpu_memory_utilization=0.8,
                    trust_remote_code=True
                )
                print("✅ Model loaded with trust_remote_code")
            except Exception as e1:
                print(f"⚠️  First attempt failed: {e1}")
                print("   Trying alternative loading method...")
                # Try without trust_remote_code
                self.llm = LLM(
                    model=base_model,
                    limit_mm_per_prompt={"video": 10},
                    dtype="bfloat16",
                    gpu_memory_utilization=0.8
                )
                print("✅ Model loaded without trust_remote_code")
        except Exception as e:
            print(f"❌ Failed to load model completely: {e}")
            print("⚠️  Model loading failed, evaluation will proceed with basic metrics only")
            self.llm = None
            self.processor = None
    
    def generate_single_prediction(self, test_item: Dict[str, Any]) -> Dict[str, Any]:
        """Generate prediction for a single test item using the actual model."""
        
        # If model not loaded, return empty prediction
        if self.llm is None or self.processor is None:
            video_path = test_item.get("video", "unknown_video.mp4")
            video_name = Path(video_path).name
            print(f"  ⚠️  Model not available, skipping prediction")
            return {
                "video_file": video_name,
                "video_path": video_path,
                "event": "Unknown",
                "description": "Model not available",
                "start_time": "0:00",
                "end_time": "0:00",
                "error": "Model loading failed"
            }
        
        video_path = test_item.get("video", "unknown_video.mp4")
        video_name = Path(video_path).name
        
        # Resolve video path
        resolved_path = Path(video_path)
        if not resolved_path.exists():
            # Try relative to test file directory
            resolved_path = self.test_file.parent.parent / video_path
            if not resolved_path.exists():
                print(f"  ⚠️  Video not found: {video_path}")
                return {
                    "video": video_path,
                    "error": "Video file not found"
                }
        
        print(f"🎬 Processing: {video_name}")
        
        try:
            # Create football analysis prompt
            football_prompt = "Analyze the following football clip. Identify all significant events including goals, cards, and shots. For each event, provide a detailed description including player, team, and jersey info, the event class, and precise start/ propagated timestamps. Output *only* a valid JSON array."
            
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
                            "video": f"file://{resolved_path.absolute()}",
                            "fps": 4,
                        }
                    ],
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
            if video_inputs is not None:
                mm_data["video"] = video_inputs
            
            # Setup sampling parameters
            sampling_params = SamplingParams(
                max_tokens=4096,
                temperature=0.6,
                top_p=0.95,
            )
            
            # Prepare inputs
            llm_inputs = {
                "prompt": prompt,
                "multi_modal_data": mm_data,
                "mm_processor_kwargs": video_kwargs,
            }
            
            # Add LoRA request if available
            if self.lora_path and self.lora_path.exists():
                lora_request = LoRARequest(
                    lora_name="football_adapter",
                    lora_int_id=1
                )
                llm_inputs["lora_request"] = lora_request
            
            # Generate prediction
            outputs = self.llm.generate([llm_inputs], sampling_params=sampling_params)
            generated_text = outputs[0].outputs[0].text.strip()
            
            # Parse JSON
            try:
                analysis = json.loads(generated_text)
                if not isinstance(analysis, list):
                    analysis = [analysis]
            except json.JSONDecodeError as e:
                print(f"  ⚠️  JSON parse error: {e}")
                analysis = [{"raw_output": generated_text}]
            
            # Extract first event as primary prediction
            primary_event = analysis[0] if analysis else {}
            
            prediction = {
                "video_file": video_name,
                "video_path": str(resolved_path),
                "event": primary_event.get("event", "Unknown"),
                "description": primary_event.get("description", generated_text),
                "start_time": primary_event.get("start_time", "0:00"),
                "end_time": primary_event.get("end_time", "0:00"),
                "all_events": analysis,
                "model_version": "cosmos-reason1-7b-lora" if self.lora_path else "cosmos-reason1-7b-base",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
            }
            
            return prediction
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return {
                "video_file": video_name,
                "video_path": str(resolved_path),
                "error": str(e)
            }
    
    def generate_all_predictions(self) -> List[Dict[str, Any]]:
        """Generate predictions for all test items."""
        print("\n🚀 Generating predictions for all test items...")
        
        predictions = []
        for i, test_item in enumerate(self.test_data, 1):
            print(f"\n[{i}/{len(self.test_data)}]")
            
            prediction = self.generate_single_prediction(test_item)
            predictions.append(prediction)
            
            if "error" not in prediction:
                print(f"  ✅ Generated prediction for {prediction['video_file']}")
        
        print(f"\n📊 Generated {len(predictions)} predictions")
        return predictions
    
    def save_predictions(self, predictions: List[Dict[str, Any]], output_file: str = "predictions.json"):
        """Save predictions to file."""
        output_path = self.output_dir / output_file
        
        with open(output_path, 'w') as f:
            json.dump(predictions, f, indent=2)
        
        print(f"💾 Predictions saved to: {output_path}")
        return output_path
    
    def save_individual_predictions(self, predictions: List[Dict[str, Any]]):
        """Save individual prediction files."""
        individual_dir = self.output_dir / "individual"
        individual_dir.mkdir(exist_ok=True)
        
        for prediction in predictions:
            video_name = prediction.get("video_file", "unknown")
            output_file = individual_dir / f"{video_name.replace('.mp4', '.json')}"
            
            with open(output_file, 'w') as f:
                json.dump(prediction, f, indent=2)
        
        print(f"📁 Individual predictions saved to: {individual_dir}")
    
    def run_prediction_generation(self):
        """Run the complete prediction generation process."""
        print("🏈 Football Video Analysis Prediction Generation")
        print("=" * 60)
        
        # Generate predictions
        predictions = self.generate_all_predictions()
        
        if not predictions:
            print("❌ No predictions generated!")
            return False
        
        # Save predictions
        self.save_predictions(predictions)
        self.save_individual_predictions(predictions)
        
        # Generate summary
        self._generate_summary(predictions)
        
        print("\n✅ Prediction generation completed successfully!")
        return True
    
    def _generate_summary(self, predictions: List[Dict[str, Any]]):
        """Generate prediction summary."""
        successful = [p for p in predictions if "error" not in p]
        
        summary = {
            "total_predictions": len(predictions),
            "successful_predictions": len(successful),
            "failed_predictions": len(predictions) - len(successful),
            "event_distribution": {},
            "model_info": {
                "model_version": "cosmos-reason1-7b-lora" if self.lora_path else "cosmos-reason1-7b-base",
                "generation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
            }
        }
        
        # Calculate event distribution
        for pred in successful:
            event = pred.get("event", "Unknown")
            summary["event_distribution"][event] = summary["event_distribution"].get(event, 0) + 1
        
        # Save summary
        summary_path = self.output_dir / "prediction_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📊 Prediction summary saved to: {summary_path}")
        
        # Print summary
        print("\n📊 PREDICTION SUMMARY:")
        print(f"  Total: {summary['total_predictions']}")
        print(f"  Successful: {summary['successful_predictions']}")
        print(f"  Failed: {summary['failed_predictions']}")
        print(f"  Event distribution: {summary['event_distribution']}")


def main():
    """Main prediction generation function."""
    parser = argparse.ArgumentParser(description="Generate Football Video Analysis Predictions")
    parser.add_argument("--test_file", type=str,
                       default="../04_dataset/validation.jsonl",
                       help="Path to validation/test.jsonl file")
    parser.add_argument("--model_path", type=str,
                       default="nvidia/Cosmos-Reason1-7B",
                       help="Path to base model")
    parser.add_argument("--lora_path", type=str,
                       default="../05_training/checkpoints/football_sft",
                       help="Path to LoRA adapter")
    parser.add_argument("--output_dir", type=str,
                       default="./results",
                       help="Directory to save predictions")
    
    args = parser.parse_args()
    
    # Create prediction generator
    generator = PredictionGenerator(
        test_file=args.test_file,
        model_path=args.model_path,
        output_dir=args.output_dir,
        lora_path=args.lora_path
    )
    
    # Run prediction generation
    success = generator.run_prediction_generation()
    
    if success:
        return 0
    else:
        return 1


if __name__ == "__main__":
    exit(main())
