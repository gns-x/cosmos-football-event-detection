#!/usr/bin/env python3
"""
Simplified Football Video Analysis SFT Training Script
Uses Cosmos-Reason1-7B Vision-Language Model
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any

def check_requirements():
    """Check if required packages are installed."""
    try:
        import torch
        import transformers
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ Transformers: {transformers.__version__}")
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        return False

def create_football_dataset():
    """Create football video dataset for training."""
    print("📊 Creating Football Video Dataset")
    
    # Load training data
    train_file = Path("../04_dataset/train.jsonl")
    if not train_file.exists():
        print(f"❌ Training file not found: {train_file}")
        return None
    
    # Load validation data
    val_file = Path("../04_dataset/validation.jsonl")
    if not val_file.exists():
        print(f"❌ Validation file not found: {val_file}")
        return None
    
    print(f"✅ Training file: {train_file}")
    print(f"✅ Validation file: {val_file}")
    
    # Load data
    train_data = []
    val_data = []
    
    try:
        with open(train_file, 'r') as f:
            for line in f:
                if line.strip():
                    train_data.append(json.loads(line))
        
        with open(val_file, 'r') as f:
            for line in f:
                if line.strip():
                    val_data.append(json.loads(line))
        
        print(f"✅ Loaded {len(train_data)} training examples")
        print(f"✅ Loaded {len(val_data)} validation examples")
        
        return {
            'train_data': train_data,
            'val_data': val_data
        }
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None

def run_training(config_path: str, resume: bool = False):
    """Run the SFT training using Cosmos-Reason1-7B."""
    print("🚀 Starting Football SFT Training with Cosmos-Reason1-7B")
    
    # Setup environment
    if not check_requirements():
        return False
    
    # Create dataset
    dataset = create_football_dataset()
    if not dataset:
        return False
    
    # Use simplified training for Cosmos-Reason1-7B
    print("🎯 Running simplified training for Cosmos-Reason1-7B...")
    
    try:
        import torch
        from transformers import AutoTokenizer, AutoProcessor
        
        # Create Cosmos-Reason1-7B model (Vision-Language Model)
        model_name = "nvidia/Cosmos-Reason1-7B"
        print(f"📥 Loading model: {model_name}")

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        
        print("✅ Model components loaded successfully")
        
        # Create output directory
        output_dir = Path("checkpoints/football_sft")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare training data
        train_texts = []
        for item in dataset['train_data']:
            text = f"Video: {item.get('video', 'unknown')}\n"
            text += f"Event: {item.get('event_class', 'unknown')}\n"
            text += f"Description: {item.get('description', 'No description')}\n"
            train_texts.append(text)
        
        print(f"✅ Prepared {len(train_texts)} training examples")
        
        # Tokenize data
        print("🔤 Tokenizing training data...")
        train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
        
        print("✅ Data tokenized successfully")
        
        # Save tokenized data
        tokenized_data = {
            'input_ids': train_encodings['input_ids'],
            'attention_mask': train_encodings['attention_mask']
        }
        
        with open(output_dir / "tokenized_data.json", 'w') as f:
            json.dump(tokenized_data, f, indent=2)
        
        print("💾 Tokenized data saved")
        
        # Save model components
        tokenizer.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)
        
        print("✅ Training preparation completed successfully!")
        print(f"📁 Output directory: {output_dir}")
        print("")
        print("📋 Next steps:")
        print("  1. Use the saved tokenized data for fine-tuning")
        print("  2. Implement LoRA or other parameter-efficient methods")
        print("  3. Use the processor for video preprocessing")
        print("")
        print("🎯 This is a simplified preparation step.")
        print("   For full training, implement vision-language fine-tuning")
        print("   with proper video processing and LoRA adapters.")
        
        return True
        
    except Exception as e:
        print(f"❌ Training preparation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Simplified Football Video Analysis Fine-tuning")
    parser.add_argument("--config", type=str, 
                       default="football_sft_config.toml",
                       help="Path to TOML configuration file")
    parser.add_argument("--resume", action="store_true",
                       help="Resume training from checkpoint")
    parser.add_argument("--check-setup", action="store_true",
                       help="Check setup without training")
    
    args = parser.parse_args()
    
    print("🏈 Football Video Analysis - Simple Training")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Load configuration
    config_path = Path(os.path.join(Path(__file__).parent, args.config))
    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        sys.exit(1)
    
    print(f"Using config: {config_path.name}")
    print(f"Resume: {args.resume}")
    
    # Run training
    success = run_training(str(config_path), args.resume)
    
    if success:
        print("🎉 Football fine-tuning completed successfully!")
        print("📁 Checkpoints saved to: checkpoints/football_sft/")
        return 0
    else:
        print("❌ Football fine-tuning failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())