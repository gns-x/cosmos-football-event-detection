#!/bin/bash

# Simple installation script for Cosmos-Reason1-7B with vLLM
# Run this in your VM after cloning the repo

echo "🚀 Installing Cosmos-Reason1-7B with vLLM..."

# Create virtual environment (avoid conda conflicts)
python3 -m venv cosmos-env
source cosmos-env/bin/activate

# Upgrade pip first
pip install --upgrade pip

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies first
pip install transformers>=4.30.0
pip install qwen-vl-utils>=0.0.1
pip install fastapi uvicorn python-multipart
pip install opencv-python pillow numpy
pip install python-dotenv aiofiles requests pydantic

# Install vLLM with specific version to avoid SQLite issues
pip install vllm==0.5.5

# Download Cosmos-Reason1-7B model (this will cache it locally)
echo "📥 Downloading Cosmos-Reason1-7B model (~14GB)..."
python -c "
from transformers import AutoTokenizer, AutoProcessor
import os

# Set environment variable for video reader
os.environ['QWEN_VL_VIDEO_READER_BACKEND'] = 'decord'

# Download tokenizer and processor
print('Downloading tokenizer...')
AutoTokenizer.from_pretrained('nvidia/Cosmos-Reason1-7B', trust_remote_code=True)

print('Downloading processor...')
AutoProcessor.from_pretrained('nvidia/Cosmos-Reason1-7B', trust_remote_code=True)

print('✅ Model components downloaded and cached locally!')
"

echo "✅ Installation complete!"
echo ""
echo "🚀 To start the backend:"
echo "source cosmos-env/bin/activate"
echo "uvicorn backend.main_simple:app --host 0.0.0.0 --port 8000"
