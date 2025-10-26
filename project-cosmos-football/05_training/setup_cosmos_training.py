#!/usr/bin/env python3
"""
Setup script for Cosmos Football Video Analysis Training
Installs required dependencies and sets up the training environment
"""

import os
import sys
import subprocess
from pathlib import Path
import argparse


def check_conda_environment():
    """Check if conda environment is activated."""
    conda_env = os.environ.get("CONDA_DEFAULT_ENV")
    if conda_env != "cosmos-football":
        print("⚠️  Warning: Not in cosmos-football conda environment")
        print("💡 Please run: conda activate cosmos-football")
        return False
    
    print(f"✅ Conda environment: {conda_env}")
    return True


def install_cosmos_dependencies():
    """Install Cosmos RL framework dependencies."""
    print("📦 Installing Cosmos RL Framework Dependencies")
    
    # Install cosmos-rl from the cookbook
    cookbook_path = Path(__file__).parent / "cosmos-cookbook"
    
    if not cookbook_path.exists():
        print(f"❌ Cosmos cookbook not found at: {cookbook_path}")
        return False
    
    try:
        # Install cosmos-rl in development mode
        cmd = ["pip", "install", "-e", str(cookbook_path)]
        print(f"📋 Installing: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Cosmos RL framework installed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def install_additional_dependencies():
    """Install additional dependencies for training."""
    print("📦 Installing Additional Dependencies")
    
    additional_deps = [
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "accelerate>=0.20.0",
        "peft>=0.4.0",
        "bitsandbytes>=0.39.0",
        "deepspeed>=0.18.0",
        "wandb>=0.22.0",
        "tensorboard>=2.20.0",
        "ray[default]>=2.50.0",
        "trl>=0.24.0"
    ]
    
    for dep in additional_deps:
        try:
            print(f"📦 Installing {dep}...")
            result = subprocess.run(
                ["pip", "install", dep], 
                check=True, 
                capture_output=True, 
                text=True
            )
            print(f"✅ {dep} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Warning: Failed to install {dep}: {e.stderr}")
    
    return True


def setup_training_directories():
    """Set up training directories."""
    print("📁 Setting up Training Directories")
    
    directories = [
        "checkpoints",
        "checkpoints/football_sft",
        "logs",
        "outputs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {directory}")
    
    return True


def verify_setup():
    """Verify the training setup."""
    print("🔍 Verifying Training Setup")
    
    # Check if required files exist
    required_files = [
        "football_sft_config.toml",
        "fine_tune.py",
        "04_dataset/train.jsonl",
        "04_dataset/validation.jsonl"
    ]
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} not found")
            return False
    
    # Check if cosmos-cookbook exists
    cookbook_path = Path("cosmos-cookbook")
    if cookbook_path.exists():
        print(f"✅ {cookbook_path}")
    else:
        print(f"❌ {cookbook_path} not found")
        return False
    
    return True


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Setup Cosmos Football Training Environment")
    parser.add_argument("--skip-install", action="store_true",
                       help="Skip dependency installation")
    parser.add_argument("--verify-only", action="store_true",
                       help="Only verify setup without installing")
    
    args = parser.parse_args()
    
    print("🏈 Cosmos Football Video Analysis Training Setup")
    print("=" * 60)
    
    # Check conda environment
    if not check_conda_environment():
        print("💡 Please activate the cosmos-football environment first")
        return 1
    
    if not args.verify_only:
        # Install dependencies
        if not args.skip_install:
            if not install_cosmos_dependencies():
                print("❌ Failed to install Cosmos RL framework")
                return 1
            
            if not install_additional_dependencies():
                print("❌ Failed to install additional dependencies")
                return 1
        
        # Setup directories
        if not setup_training_directories():
            print("❌ Failed to setup training directories")
            return 1
    
    # Verify setup
    if not verify_setup():
        print("❌ Setup verification failed")
        return 1
    
    print("=" * 60)
    print("✅ Cosmos Football Training Setup Complete!")
    print("")
    print("🚀 Ready to start training:")
    print("   python3 05_training/fine_tune.py --check-setup")
    print("   python3 05_training/fine_tune.py")
    print("")
    print("📋 Training configuration: football_sft_config.toml")
    print("📁 Output directory: checkpoints/football_sft/")
    
    return 0


if __name__ == "__main__":
    exit(main())
