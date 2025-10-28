#!/bin/bash
# Fix deprecated package warnings in Cosmos environment

echo "🔧 Fixing deprecated package warnings..."
echo "=========================================="

# Activate the environment
source cosmos-env/bin/activate

echo "📦 Updating deprecated packages..."

# Fix pynvml deprecation warning
echo "1. Installing nvidia-ml-py (replaces pynvml)..."
pip uninstall pynvml -y 2>/dev/null || true
pip install nvidia-ml-py

# Fix video processing deprecation warnings
echo "2. Installing torchcodec for modern video processing..."
pip install torchcodec

# Update transformers to latest version
echo "3. Updating transformers to latest version..."
pip install --upgrade transformers

# Update torchvision to latest version
echo "4. Updating torchvision..."
pip install --upgrade torchvision

# Install additional video processing dependencies
echo "5. Installing additional video processing dependencies..."
pip install av opencv-python-headless

# Update other core packages
echo "6. Updating core packages..."
pip install --upgrade torch
pip install --upgrade peft

echo ""
echo "✅ Package updates completed!"
echo ""
echo "📊 Updated packages:"
echo "  - nvidia-ml-py (replaces pynvml)"
echo "  - torchcodec (modern video processing)"
echo "  - transformers (latest version)"
echo "  - torchvision (latest version)"
echo "  - av, opencv-python-headless (video processing)"
echo ""
echo "🎯 These updates should eliminate the deprecation warnings."
