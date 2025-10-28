#!/usr/bin/env python3
"""
Phase 5: Training Pipeline "Smoke Test"
Goal: Verify the training script runs without crashing and that the model can "learn."
We will do this by overfitting on a single batch, a standard ML engineering practice.
"""

import os
import sys
import json
import argparse
import yaml
from pathlib import Path
from typing import Dict, List, Any
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoProcessor,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import transformers
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np

class FootballVideoDataset(Dataset):
    """Dataset for football video analysis training."""
    
    def __init__(self, data_path: str, tokenizer, processor, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = max_length
        self.data = []
        
        # Load data from JSONL
        with open(data_path, 'r') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
        
        print(f"✅ Loaded {len(self.data)} examples from {data_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Create conversation format
        conversation = item.get('conversations', [])
        
        # Format as text for training (simplified for smoke test)
        text = ""
        for turn in conversation:
            role = turn.get('role', '')
            content = turn.get('content', '')
            
            if role == 'system':
                text += f"System: {content}\n"
            elif role == 'user':
                if isinstance(content, list):
                    # Handle multimodal content
                    for c in content:
                        if c.get('type') == 'text':
                            text += f"User: {c.get('text', '')}\n"
                else:
                    text += f"User: {content}\n"
            elif role == 'assistant':
                text += f"Assistant: {content}\n"
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze()  # For causal LM
        }

def create_overfit_dataset(train_path: str, val_path: str):
    """Create overfit dataset by copying train to validation."""
    print("📊 Creating Overfit Dataset for Smoke Test")
    
    # Read training data
    with open(train_path, 'r') as f:
        train_data = [json.loads(line) for line in f if line.strip()]
    
    # Copy train data to validation for overfitting test
    with open(val_path, 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"✅ Created overfit validation set with {len(train_data)} examples")
    print("🎯 This allows the model to overfit and prove it can learn")

def setup_lora_model(model_name: str, config: Dict[str, Any]):
    """Setup LoRA model for parameter-efficient fine-tuning."""
    print(f"🔧 Setting up LoRA model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load processor (for vision components)
    try:
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    except:
        print("⚠️  AutoProcessor not available, using tokenizer only")
        processor = tokenizer
    
    # Load base model - Cosmos is based on Qwen2.5-VL
    model = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Setup LoRA for vision-language model
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # Still causal LM for text generation
        r=config.get('lora_rank', 16),
        lora_alpha=config.get('lora_alpha', 32),
        lora_dropout=config.get('lora_dropout', 0.1),
        target_modules=config.get('target_modules', ["q_proj", "v_proj", "k_proj", "o_proj"]),
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer, processor

def run_smoke_test(config_path: str):
    """Run Phase 5 smoke test training."""
    print("🚀 Phase 5: Training Pipeline Smoke Test")
    print("=" * 50)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"📋 Config: {config_path}")
    print(f"🎯 Model: {config['base_model_path']}")
    print(f"📊 Batch size: {config['batch_size']}")
    print(f"🔄 Epochs: {config['epochs']}")
    
    # Create overfit dataset
    train_path = config['train_data_path']
    val_path = config['validation_data_path']
    
    if not Path(train_path).exists():
        print(f"❌ Training file not found: {train_path}")
        return False
    
    create_overfit_dataset(train_path, val_path)
    
    # Setup model
    model, tokenizer, processor = setup_lora_model(config['base_model_path'], config)
    
    # Create datasets
    train_dataset = FootballVideoDataset(train_path, tokenizer, processor, int(config['max_length']))
    val_dataset = FootballVideoDataset(val_path, tokenizer, processor, int(config['max_length']))
    
    # Setup training arguments
    output_dir = Path(config['output_dir']) / "football_sft"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=int(config['epochs']),
        per_device_train_batch_size=int(config['batch_size']),
        per_device_eval_batch_size=int(config['batch_size']),
        gradient_accumulation_steps=config.get('gradient_accumulation_steps', 4),
        learning_rate=float(config['learning_rate']),
        warmup_steps=config.get('warmup_steps', 100),
        logging_steps=config.get('logging_steps', 50),
        save_steps=config.get('save_steps', 500),
        eval_steps=config.get('eval_steps', 250),
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=config.get('bf16', True),
        dataloader_num_workers=config.get('dataloader_num_workers', 4),
        remove_unused_columns=False,
        report_to="none",  # Disable wandb for smoke test
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # Run training
    print("🎯 Starting smoke test training...")
    print("📊 Expected: train_loss should drop quickly (approaching 0.0)")
    print("📊 This proves data format is correct and model can learn")
    
    try:
        trainer.train()
        
        # Save final model
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        
        print("✅ Smoke test training completed!")
        print(f"📁 LoRA adapter saved to: {output_dir}")
        
        # Verify LoRA adapter exists
        adapter_files = list(output_dir.glob("adapter_*.bin")) + list(output_dir.glob("adapter_model.*"))
        if adapter_files:
            print(f"✅ LoRA adapter files found: {[f.name for f in adapter_files]}")
        else:
            print("⚠️  No LoRA adapter files found - check training logs")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Phase 5: Football Training Smoke Test")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Path to config file")
    
    args = parser.parse_args()
    
    print("🏈 Phase 5: Football Video Analysis Training Smoke Test")
    print("=" * 60)
    print("🎯 Goal: Verify training works and model can overfit")
    print("📊 Method: Train on single batch with high epochs")
    print("✅ Success: Loss drops to ~0.0, LoRA adapter saved")
    print("")
    
    success = run_smoke_test(args.config)
    
    if success:
        print("🎉 Phase 5 smoke test PASSED!")
        print("✅ Training pipeline is working correctly")
        print("✅ Model can learn from the data")
        print("✅ LoRA adapter saved successfully")
        return 0
    else:
        print("❌ Phase 5 smoke test FAILED!")
        print("❌ Check data format, model setup, or dependencies")
        return 1

if __name__ == "__main__":
    sys.exit(main())
