#!/usr/bin/env python3
"""
Football Video Analysis SFT Training Script
Based on Cosmos RL framework with LoRA fine-tuning
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any

import cosmos_rl.launcher.worker_entry
import cosmos_rl.policy.config
import pydantic
import toml
import torch.utils.data
from cosmos_reason1_utils.text import create_conversation
from cosmos_reason1_utils.vision import VisionConfig
from cosmos_rl.utils.logging import logger


class FootballDatasetConfig(pydantic.BaseModel):
    annotation_path: str = pydantic.Field()
    """Dataset annotation path."""
    media_path: str = pydantic.Field(default="")
    """Dataset media path."""
    system_prompt: str = pydantic.Field(default="")
    """System prompt for football video analysis."""


class FootballConfig(pydantic.BaseModel):
    dataset: FootballDatasetConfig = pydantic.Field()
    """Dataset config."""

    vision: VisionConfig = pydantic.Field(
        default=VisionConfig(
            fps=4,  # 4 FPS as required by Cosmos-Reason1-7B
            max_pixels=81920,
            max_video_frames=10,
            max_image_frames=10,
        )
    )
    """Vision config for football videos."""


class FootballDataset(torch.utils.data.Dataset):
    """Football video dataset for SFT training."""
    
    def __init__(self, config: FootballConfig):
        self.config = config
        self.data = self._load_data()
        
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load football video data from JSONL files."""
        data = []
        
        # Load training data
        train_file = Path(self.config.dataset.annotation_path)
        if train_file.exists():
            with open(train_file, 'r') as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        data.append(item)
        
        logger.info(f"Loaded {len(data)} football video examples")
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single training example."""
        item = self.data[idx]
        
        # Create conversation format for Cosmos
        conversation = [
            {
                "role": "system",
                "content": self.config.dataset.system_prompt
            },
            {
                "role": "user", 
                "content": [
                    {
                        "type": "text",
                        "text": item["prompt"]
                    },
                    {
                        "type": "video",
                        "video": str(Path(self.config.dataset.media_path) / item["video"]),
                        "fps": 4  # 4 FPS as required
                    }
                ]
            },
            {
                "role": "assistant",
                "content": item["completion"]
            }
        ]
        
        # Create the conversation string
        conversation_str = create_conversation(conversation)
        
        return {
            "conversations": conversation_str,
            "video_path": item["video"]
        }


def create_football_config(config_path: str) -> FootballConfig:
    """Create football-specific configuration."""
    config_dict = toml.load(config_path)
    
    # Extract custom config
    custom_config = config_dict.get("custom", {})
    
    # Create dataset config
    dataset_config = FootballDatasetConfig(
        annotation_path=custom_config.get("dataset", {}).get("annotation_path", ""),
        media_path=custom_config.get("dataset", {}).get("media_path", ""),
        system_prompt=custom_config.get("dataset", {}).get("system_prompt", "")
    )
    
    # Create vision config
    vision_config = VisionConfig(
        fps=custom_config.get("vision", {}).get("fps", 4),
        max_pixels=custom_config.get("vision", {}).get("max_pixels", 81920),
        max_video_frames=custom_config.get("vision", {}).get("max_video_frames", 10),
        max_image_frames=custom_config.get("vision", {}).get("max_image_frames", 10)
    )
    
    return FootballConfig(
        dataset=dataset_config,
        vision=vision_config
    )


def main():
    """Main training function for football video analysis."""
    parser = argparse.ArgumentParser(description="Football Video Analysis SFT Training")
    parser.add_argument("--config", type=str, required=True, 
                       help="Path to TOML configuration file")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/football_sft",
                       help="Output directory for checkpoints")
    parser.add_argument("--resume", action="store_true", 
                       help="Resume training from checkpoint")
    
    args = parser.parse_args()
    
    # Load configuration
    config = create_football_config(args.config)
    
    logger.info("Starting Football Video Analysis SFT Training")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Resume: {args.resume}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dataset
    dataset = FootballDataset(config)
    logger.info(f"Dataset created with {len(dataset)} examples")
    
    # Start training
    try:
        cosmos_rl.launcher.worker_entry.main()
        logger.info("Football SFT training completed successfully!")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
