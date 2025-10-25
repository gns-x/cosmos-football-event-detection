#!/usr/bin/env python3
"""
Azure A100 Training Script for Cosmos Football Video Analysis
Optimized for Standard_NC24ads_A100_v4 instances
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List
import argparse
import toml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import training components
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
    DataCollatorForLanguageModeling, BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import wandb
from loguru import logger

class AzureCosmosTrainer:
    """Azure A100 optimized trainer for Cosmos Football Analysis."""
    
    def __init__(self, config_path: str):
        """Initialize trainer with Azure-optimized configuration."""
        self.config = self.load_config(config_path)
        self.setup_logging()
        self.setup_azure_environment()
        
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load training configuration."""
        with open(config_path, 'r') as f:
            config = toml.load(f)
        return config
    
    def setup_logging(self):
        """Setup logging for Azure environment."""
        log_level = self.config.get('logging', {}).get('log_level', 'INFO')
        logger.remove()
        logger.add(
            sys.stdout,
            level=log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        )
        logger.add(
            "logs/azure_training_{time:YYYY-MM-DD_HH-mm-ss}.log",
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
        )
    
    def setup_azure_environment(self):
        """Setup Azure-specific environment variables."""
        azure_config = self.config.get('azure', {})
        
        # Set Azure-specific environment variables
        os.environ['AZURE_VM'] = 'true'
        os.environ['GPU_COUNT'] = str(azure_config.get('gpu_count', 1))
        os.environ['GPU_MEMORY'] = str(azure_config.get('gpu_memory', 80))
        
        # Setup Weights & Biases for Azure
        if self.config.get('logging', {}).get('use_wandb', False):
            wandb.init(
                project=self.config['logging']['wandb_project'],
                entity=self.config['logging'].get('wandb_entity', 'cosmos-football'),
                config=self.config,
                name=f"cosmos-football-azure-{int(time.time())}"
            )
            logger.info("Weights & Biases initialized for Azure training")
    
    def check_azure_environment(self):
        """Check Azure A100 environment."""
        logger.info("🔧 Checking Azure A100 environment...")
        
        # Check GPU availability
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available on Azure VM")
        
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        
        logger.info(f"GPU Count: {gpu_count}")
        logger.info(f"GPU Name: {gpu_name}")
        
        # Check memory
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU Memory: {gpu_memory:.1f} GB")
        
        # Verify Azure VM type
        expected_memory = self.config['azure']['gpu_memory']
        if gpu_memory < expected_memory * 0.9:  # Allow 10% tolerance
            logger.warning(f"Expected {expected_memory}GB GPU memory, got {gpu_memory:.1f}GB")
        
        logger.info("✅ Azure A100 environment verified")
    
    def setup_model(self):
        """Setup model with Azure-optimized configuration."""
        logger.info("🚀 Setting up Cosmos model for Azure A100...")
        
        model_config = self.config['model']
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_config['base_model_path'],
            trust_remote_code=True
        )
        
        # Setup quantization for A100
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_config['base_model_path'],
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
        # Setup LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config['lora']['lora_rank'],
            lora_alpha=self.config['lora']['lora_alpha'],
            lora_dropout=self.config['lora']['lora_dropout'],
            target_modules=self.config['lora']['target_modules']
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        logger.info("✅ Model setup completed for Azure A100")
        return model, tokenizer
    
    def setup_training_args(self):
        """Setup training arguments optimized for Azure A100."""
        logger.info("⚙️  Setting up training arguments for Azure A100...")
        
        training_config = self.config['training']
        output_config = self.config['output']
        hardware_config = self.config['hardware']
        
        training_args = TrainingArguments(
            output_dir=output_config['output_dir'],
            num_train_epochs=training_config['num_train_epochs'],
            per_device_train_batch_size=training_config['per_device_train_batch_size'],
            per_device_eval_batch_size=training_config['per_device_train_batch_size'],
            gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
            learning_rate=training_config['learning_rate'],
            weight_decay=training_config['weight_decay'],
            warmup_steps=training_config['warmup_steps'],
            logging_steps=output_config['logging_steps'],
            save_steps=output_config['save_steps'],
            eval_steps=output_config['eval_steps'],
            save_total_limit=output_config['save_total_limit'],
            load_best_model_at_end=output_config['load_best_model_at_end'],
            metric_for_best_model=output_config['metric_for_best_model'],
            greater_is_better=output_config['greater_is_better'],
            fp16=hardware_config['fp16'],
            bf16=hardware_config['bf16'],
            dataloader_num_workers=hardware_config['dataloader_num_workers'],
            remove_unused_columns=hardware_config['remove_unused_columns'],
            report_to=["wandb", "tensorboard"] if self.config['logging']['use_wandb'] else ["tensorboard"],
            run_name=f"cosmos-football-azure-{int(time.time())}",
            seed=self.config['other']['seed'],
            disable_tqdm=self.config['other']['disable_tqdm']
        )
        
        # Add DeepSpeed configuration if enabled
        if self.config['deepspeed']['use_deepspeed']:
            training_args.deepspeed = self.config['deepspeed']['deepspeed_config']
            logger.info("✅ DeepSpeed configuration added for Azure A100")
        
        logger.info("✅ Training arguments configured for Azure A100")
        return training_args
    
    def load_dataset(self):
        """Load and prepare dataset for training."""
        logger.info("📊 Loading dataset for Azure training...")
        
        data_config = self.config['data']
        
        # Load training data
        train_data = []
        with open(data_config['train_data_path'], 'r') as f:
            for line in f:
                train_data.append(json.loads(line.strip()))
        
        # Load validation data
        val_data = []
        with open(data_config['validation_data_path'], 'r') as f:
            for line in f:
                val_data.append(json.loads(line.strip()))
        
        logger.info(f"Training examples: {len(train_data)}")
        logger.info(f"Validation examples: {len(val_data)}")
        
        return train_data, val_data
    
    def train(self):
        """Execute training on Azure A100."""
        logger.info("🚀 Starting Cosmos training on Azure A100...")
        
        # Check environment
        self.check_azure_environment()
        
        # Setup model and tokenizer
        model, tokenizer = self.setup_model()
        
        # Setup training arguments
        training_args = self.setup_training_args()
        
        # Load dataset
        train_data, val_data = self.load_dataset()
        
        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=val_data,
            data_collator=data_collator,
            tokenizer=tokenizer
        )
        
        # Start training
        logger.info("🎯 Starting training on Azure A100...")
        start_time = time.time()
        
        trainer.train()
        
        training_time = time.time() - start_time
        logger.info(f"✅ Training completed in {training_time:.2f} seconds")
        
        # Save final model
        trainer.save_model()
        logger.info("✅ Model saved to checkpoints/")
        
        # Log final metrics
        if hasattr(trainer, 'state') and trainer.state.log_history:
            final_metrics = trainer.state.log_history[-1]
            logger.info(f"Final metrics: {final_metrics}")
        
        logger.info("🎉 Azure A100 training completed successfully!")
        
        return trainer

def main():
    """Main training function for Azure A100."""
    parser = argparse.ArgumentParser(description="Azure A100 Training for Cosmos Football Analysis")
    parser.add_argument("--config", type=str, default="azure_training_config.toml",
                       help="Path to training configuration file")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = AzureCosmosTrainer(args.config)
    
    # Execute training
    trainer.train()
    
    logger.info("🎉 Azure A100 training pipeline completed!")

if __name__ == "__main__":
    main()
