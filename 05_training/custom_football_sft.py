#!/usr/bin/env python3
"""
Football Video Analysis SFT Training with Cosmos RL (2025 Best Practices)
Uses NVIDIA Cosmos RL framework for professional SFT training
"""

import argparse
import json
import os
from pathlib import Path

import cosmos_rl.launcher.worker_entry
import cosmos_rl.policy.config
import pydantic
import toml
import torch.utils.data
from cosmos_reason1_utils.text import create_conversation
from cosmos_reason1_utils.vision import VisionConfig
from cosmos_rl.utils.logging import logger


class CustomDatasetConfig(pydantic.BaseModel):
    annotation_path: str = pydantic.Field()
    media_path: str = pydantic.Field(default="")
    system_prompt: str = pydantic.Field(default="")


class CustomConfig(pydantic.BaseModel):
    dataset: CustomDatasetConfig = pydantic.Field()
    
    vision: VisionConfig = pydantic.Field(
        default=VisionConfig(
            fps=4,  # 4 FPS for Cosmos-Reason1-7B
            max_pixels=81920,
        )
    )


class FootballDataset(torch.utils.data.Dataset):
    """Football video dataset for Cosmos RL SFT training."""
    
    def __init__(
        self,
        config: cosmos_rl.policy.config.Config,
        custom_config: CustomConfig,
    ):
        # Load LLaVA format annotations
        self.annotation = json.load(open(custom_config.dataset.annotation_path))
        self.media_path = custom_config.dataset.media_path
        self.system_prompt = custom_config.dataset.system_prompt
        self.config = config
        self.custom_config = custom_config
        self.vision_kwargs = custom_config.vision.model_dump(exclude_none=True)
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, idx: int) -> list[dict]:
        """Return conversation in Cosmos format."""
        sample = self.annotation[idx]
        
        # Extract conversation
        user_prompt = sample["conversations"][0]["value"]
        response = sample["conversations"][1]["value"]
        
        # Get video path
        videos = sample.get("video", None)
        if videos and isinstance(videos, str):
            videos = [videos]
        
        # Join with media path if provided
        if self.media_path != "" and videos:
            videos = [Path(self.media_path) / vid for vid in videos]
            videos = [str(v) for v in videos]
        
        # Create conversation with Cosmos utilities
        conversations = create_conversation(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            response=response,
            images=None,
            videos=videos,
            vision_kwargs=self.vision_kwargs,
        )
        
        return conversations


def main():
    """Main training function using Cosmos RL."""
    parser = argparse.ArgumentParser(description="Football Video Analysis SFT with Cosmos RL")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    
    args = parser.parse_known_args()[0]
    
    print("🚀 Starting Football SFT Training with Cosmos RL")
    print(f"📋 Config: {args.config}")
    
    # Load config
    with open(args.config) as f:
        config_kwargs = toml.load(f)
    
    config = cosmos_rl.policy.config.Config.from_dict(config_kwargs)
    custom_config = CustomConfig.model_validate(config_kwargs["custom"])
    
    # Check if controller
    role = os.environ.get("COSMOS_ROLE")
    is_controller = role == "Controller"
    
    if is_controller:
        output_dir = Path(config.train.output_dir).resolve().parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config_kwargs["custom"] = custom_config.model_dump()
        config_path = output_dir / "config.toml"
        config_path.write_text(toml.dumps(config_kwargs))
        logger.info(f"✅ Saved config to {config_path}")
    
    # Load dataset
    print("📊 Loading football dataset...")
    dataset = FootballDataset(
        config=config,
        custom_config=custom_config,
    )
    
    print(f"✅ Loaded {len(dataset)} training examples")
    print(f"📹 Sample conversation: {dataset[0]}")
    
    # Launch Cosmos RL training
    print("🎯 Launching Cosmos RL training...")
    cosmos_rl.launcher.worker_entry.main(dataset=dataset)
    
    print("✅ Training completed!")


if __name__ == "__main__":
    main()

