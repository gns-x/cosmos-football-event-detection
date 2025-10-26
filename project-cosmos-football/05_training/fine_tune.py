#!/usr/bin/env python3
"""
Football Video Analysis Fine-tuning Script
Adapted from Cosmos Cookbook for football video analysis
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any

# Add cosmos-cookbook to path
cookbook_path = Path(__file__).parent / "cosmos-cookbook"
sys.path.insert(0, str(cookbook_path))

try:
    import cosmos_rl.launcher.worker_entry
    import cosmos_rl.policy.config
    from cosmos_reason1_utils.text import create_conversation
    from cosmos_reason1_utils.vision import VisionConfig
    from cosmos_rl.utils.logging import logger
    print("✅ Cosmos RL framework imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Please ensure Cosmos RL framework is installed")
    print("💡 Try: pip install git+https://github.com/nvidia-cosmos/cosmos-rl.git --no-deps")
    sys.exit(1)


class FootballSFTConfig:
    """Football Video Analysis SFT Configuration."""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.load_config()
    
    def load_config(self):
        """Load configuration from TOML file."""
        import toml
        
        try:
            self.config = toml.load(self.config_path)
            print(f"✅ Loaded configuration from {self.config_path}")
        except Exception as e:
            print(f"❌ Error loading config: {e}")
            sys.exit(1)
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self.config.get("train", {})
    
    def get_policy_config(self) -> Dict[str, Any]:
        """Get policy configuration."""
        return self.config.get("policy", {})
    
    def get_dataset_config(self) -> Dict[str, Any]:
        """Get dataset configuration."""
        return self.config.get("custom", {}).get("dataset", {})


def create_football_dataset(config: FootballSFTConfig):
    """Create football video dataset for training."""
    print("📊 Creating Football Video Dataset")
    
    # Load training data
    train_file = Path("04_dataset/train.jsonl")
    if not train_file.exists():
        print(f"❌ Training file not found: {train_file}")
        return None
    
    # Load validation data
    val_file = Path("04_dataset/validation.jsonl")
    if not val_file.exists():
        print(f"❌ Validation file not found: {val_file}")
        return None
    
    print(f"✅ Training file: {train_file}")
    print(f"✅ Validation file: {val_file}")
    
    return {
        "train_file": str(train_file),
        "val_file": str(val_file)
    }


def setup_training_environment():
    """Set up the training environment."""
    print("🔧 Setting up Training Environment")
    
    # Set environment variables
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
    os.environ["PYTHONPATH"] = str(Path(__file__).parent / "cosmos-cookbook")
    
    # Create output directory
    output_dir = Path("checkpoints/football_sft")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"✅ Environment setup complete")
    print(f"📁 Output directory: {output_dir}")


def run_training(config_path: str, resume: bool = False):
    """Run the SFT training."""
    print("🚀 Starting Football SFT Training")
    
    # Setup environment
    setup_training_environment()
    
    # Load configuration
    config = FootballSFTConfig(config_path)
    
    # Create dataset
    dataset = create_football_dataset(config)
    if not dataset:
        return False
    
    # Prepare training command
    cmd = [
        "python", "-m", "cosmos_rl.launcher.worker_entry",
        "--config", config_path
    ]
    
    if resume:
        cmd.append("--resume")
    
    print(f"📋 Training command: {' '.join(cmd)}")
    
    try:
        # Change to training directory
        os.chdir(Path(__file__).parent)
        
        # Run training
        import subprocess
        result = subprocess.run(cmd, check=True)
        
        print("✅ Training completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Football Video Analysis Fine-tuning")
    parser.add_argument("--config", type=str, 
                       default="football_sft_config.toml",
                       help="Path to TOML configuration file")
    parser.add_argument("--resume", action="store_true",
                       help="Resume training from checkpoint")
    parser.add_argument("--check-setup", action="store_true",
                       help="Check setup without training")
    
    args = parser.parse_args()
    
    print("🏈 Football Video Analysis Fine-tuning")
    print("=" * 50)
    
    # Check if config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        return 1
    
    print(f"📋 Using config: {config_path}")
    print(f"🔄 Resume: {args.resume}")
    
    if args.check_setup:
        print("🔍 Checking setup...")
        config = FootballSFTConfig(args.config)
        dataset = create_football_dataset(config)
        if dataset:
            print("✅ Setup check passed!")
            return 0
        else:
            print("❌ Setup check failed!")
            return 1
    
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
    exit(main())