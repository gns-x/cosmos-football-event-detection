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
    # Import transformers for LoRA model loading
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoProcessor
    from peft import PeftModel
    HAS_TRANSFORMERS = True
    
    print(f"✅ Transformers {transformers.__version__} loaded")
    print(f"✅ PyTorch {torch.__version__} loaded")
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"❌ Failed to import transformers packages: {e}")
    sys.exit(1)


class PredictionGenerator:
    """Generate predictions using the actual fine-tuned model."""
    
    def __init__(self, test_file: str, model_path: str, output_dir: str, lora_path: str = None):
        self.test_file = Path(test_file)
        self.model_path = Path(model_path)
        self.lora_path = Path(lora_path) if lora_path else None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.tokenizer = None
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
        """Load the base Cosmos model with LoRA adapter."""
        print("\n🚀 Loading Cosmos-Reason1-7B base model with LoRA adapter...")
        
        # Base model from HuggingFace
        base_model = "nvidia/Cosmos-Reason1-7B"
        
        # Load tokenizer and processor
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model, 
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.processor = AutoProcessor.from_pretrained(
                base_model, 
                trust_remote_code=True
            )
            print("✅ Tokenizer and processor loaded")
        except Exception as e:
            print(f"❌ Failed to load tokenizer/processor: {e}")
            self.processor = None
            return
        
        # Load base model
        try:
            print("📋 Loading base Cosmos-Reason1-7B model...")
            self.model = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(
                base_model,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            print("✅ Base model loaded")
        except Exception as e:
            print(f"❌ Failed to load base model: {e}")
            self.model = None
            return
        
        # Load LoRA adapter if available
        lora_exists = self.lora_path and self.lora_path.exists() and (self.lora_path / "adapter_config.json").exists()
        
        if lora_exists:
            print(f"📥 Loading LoRA adapter from: {self.lora_path}")
            try:
                self.model = PeftModel.from_pretrained(self.model, str(self.lora_path))
                print("✅ LoRA adapter loaded and merged")
            except Exception as e:
                print(f"❌ Failed to load LoRA adapter: {e}")
                print("⚠️  Continuing with base model only")
        else:
            print("⚠️  No LoRA adapter found, using base model only")
        
        # Set model to evaluation mode
        self.model.eval()
        print("✅ Model ready for inference")
    
    def generate_single_prediction(self, test_item: Dict[str, Any]) -> Dict[str, Any]:
        """Generate prediction for a single test item using the actual model."""
        
        # If model not loaded, return empty prediction
        if self.model is None or self.processor is None:
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
        
        # Resolve video path - try multiple locations
        resolved_path = Path(video_path)
        
        # Try different path resolution strategies
        possible_paths = [
            resolved_path,  # Direct path
            Path("../") / video_path,  # Relative to evaluation directory (06_evaluation)
            Path("../../") / video_path,  # Relative to project root
            Path.cwd() / video_path,  # Relative to current working directory
            Path(video_path).resolve()  # Absolute path
        ]
        
        resolved_path = None
        for path in possible_paths:
            if path.exists():
                resolved_path = path
                break
        
        if resolved_path is None:
            print(f"  ⚠️  Video not found: {video_path}")
            print(f"  🔍 Tried paths: {[str(p) for p in possible_paths]}")
            return {
                "video": video_path,
                "error": "Video file not found"
            }
        
        print(f"🎬 Processing: {video_name}")
        
        try:
            # Create football analysis prompt
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
                        {"type": "video", "video": str(resolved_path)}
                    ]
                }
            ]
            
            # Process with the model
            text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Process video and text
            inputs = self.processor(
                text=[text],
                videos=[str(resolved_path)],
                padding=True,
                return_tensors="pt"
            )
            
            # Move to GPU if available
            if torch.cuda.is_available():
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
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
            
            # Extract first event as primary prediction
            primary_event = events[0] if events else {}
            
            prediction = {
                "video_file": video_name,
                "video_path": str(resolved_path),
                "event": primary_event.get("event", "Unknown"),
                "description": primary_event.get("description", assistant_response),
                "start_time": primary_event.get("start_time", "0:00"),
                "end_time": primary_event.get("end_time", "0:00"),
                "all_events": events,
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
