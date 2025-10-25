#!/usr/bin/env python3
"""
Football Video Analysis SFT Training Launcher
Uses Cosmos RL framework for LoRA fine-tuning
"""

import os
import sys
import subprocess
from pathlib import Path
import argparse


def setup_environment():
    """Set up the training environment."""
    print("🔧 Setting up Football SFT Training Environment")
    
    # Set environment variables
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
    os.environ["PYTHONPATH"] = str(Path(__file__).parent / "cosmos-cookbook")
    
    # Add cosmos-cookbook to Python path
    cookbook_path = Path(__file__).parent / "cosmos-cookbook"
    if str(cookbook_path) not in sys.path:
        sys.path.insert(0, str(cookbook_path))
    
    print(f"✅ Environment setup complete")
    print(f"📁 Cookbook path: {cookbook_path}")


def check_dependencies():
    """Check if required dependencies are available."""
    print("🔍 Checking dependencies...")
    
    try:
        import cosmos_rl
        print("✅ cosmos_rl available")
    except ImportError:
        print("❌ cosmos_rl not found. Please install Cosmos RL framework")
        return False
    
    try:
        import cosmos_reason1_utils
        print("✅ cosmos_reason1_utils available")
    except ImportError:
        print("❌ cosmos_reason1_utils not found. Please install Cosmos Reason1 utils")
        return False
    
    try:
        import torch
        print(f"✅ PyTorch available: {torch.__version__}")
    except ImportError:
        print("❌ PyTorch not found. Please install PyTorch")
        return False
    
    return True


def launch_training(config_path: str, output_dir: str, resume: bool = False):
    """Launch the SFT training."""
    print("🚀 Launching Football SFT Training")
    
    # Prepare command
    cmd = [
        "python", "-m", "cosmos_rl.launcher.worker_entry",
        "--config", config_path,
        "--output_dir", output_dir
    ]
    
    if resume:
        cmd.append("--resume")
    
    print(f"📋 Command: {' '.join(cmd)}")
    
    # Change to training directory
    training_dir = Path(__file__).parent
    os.chdir(training_dir)
    
    try:
        # Run training
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("✅ Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False


def main():
    """Main launcher function."""
    parser = argparse.ArgumentParser(description="Football Video Analysis SFT Training Launcher")
    parser.add_argument("--config", type=str, 
                       default="football_sft_config.toml",
                       help="Path to TOML configuration file")
    parser.add_argument("--output_dir", type=str, 
                       default="./checkpoints/football_sft",
                       help="Output directory for checkpoints")
    parser.add_argument("--resume", action="store_true",
                       help="Resume training from checkpoint")
    parser.add_argument("--check-deps", action="store_true",
                       help="Check dependencies only")
    
    args = parser.parse_args()
    
    print("🏈 Football Video Analysis SFT Training Launcher")
    print("=" * 60)
    
    # Setup environment
    setup_environment()
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Dependency check failed. Please install missing dependencies.")
        return 1
    
    if args.check_deps:
        print("✅ All dependencies available!")
        return 0
    
    # Check if config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        return 1
    
    print(f"📋 Using config: {config_path}")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"🔄 Resume: {args.resume}")
    
    # Launch training
    success = launch_training(
        config_path=str(config_path),
        output_dir=args.output_dir,
        resume=args.resume
    )
    
    if success:
        print("🎉 Football SFT training completed successfully!")
        print(f"📁 Checkpoints saved to: {args.output_dir}")
        return 0
    else:
        print("❌ Football SFT training failed!")
        return 1


if __name__ == "__main__":
    exit(main())
