#!/bin/bash

# Activation script for the Cosmos Football project
# This script activates the conda environment and sets up the project

echo "🚀 Activating Cosmos Football Environment..."

# Initialize conda for this shell session
eval "$(conda shell.bash hook)"

# Activate conda environment
conda activate cosmos-football

# Check if activation was successful
if [ $? -eq 0 ]; then
    echo "✅ Environment activated successfully!"
    echo "📦 Python version: $(python --version)"
    echo "🔥 PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
    echo "🎯 CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
    echo ""
    echo "📁 Project structure:"
    echo "├── 01_data_collection/     # Video download scripts"
    echo "├── 02_preprocessing/        # Video processing with FFmpeg"
    echo "├── 03_annotation/          # Ground truth annotations"
    echo "├── 04_dataset/             # Dataset creation and splits"
    echo "├── 05_training/            # Model fine-tuning"
    echo "├── 06_evaluation/          # Model evaluation"
    echo "└── 07_inference/           # Production inference"
    echo ""
    echo "🔧 To deactivate: conda deactivate"
else
    echo "❌ Failed to activate environment"
    echo "💡 Try running: conda create -n cosmos-football python=3.11"
fi
