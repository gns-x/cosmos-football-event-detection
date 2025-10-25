#!/bin/bash
# Azure A100 Quick Start Script
# One-command deployment for Cosmos Football Video Analysis

set -e

echo "🚀 Azure A100 Quick Start - Cosmos Football Video Analysis"
echo "=================================================================="
echo "⚡ This script will set up everything for Azure A100 VM deployment"
echo ""

# Check if running on Azure
if [[ -f /etc/azure/azure.conf ]]; then
    echo "✅ Running on Azure VM"
else
    echo "⚠️  Not detected as Azure VM - continuing anyway"
fi

# Check GPU availability
echo "🔧 Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    echo "✅ NVIDIA GPU detected"
else
    echo "❌ NVIDIA GPU not found - installing drivers..."
    sudo apt-get update
    sudo apt-get install -y nvidia-driver-535
    echo "⚠️  NVIDIA drivers installed - reboot may be required"
fi

# Update system
echo "📦 Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install system dependencies
echo "📦 Installing system dependencies..."
sudo apt-get install -y \
    python3 python3-pip python3-venv python3-dev \
    git wget curl ffmpeg build-essential cmake \
    pkg-config libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libv4l-dev libxvidcore-dev libx264-dev \
    libgtk-3-dev libatlas-base-dev gfortran yt-dlp

# Create Python environment
echo "🐍 Setting up Python environment..."
python3 -m venv cosmos-env
source cosmos-env/bin/activate
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA
echo "🔥 Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
echo "📦 Installing core dependencies..."
pip install \
    transformers>=4.30.0 \
    vllm>=0.6.0 \
    qwen_vl_utils>=0.0.1 \
    accelerate>=0.20.0 \
    peft>=0.4.0 \
    bitsandbytes>=0.39.0 \
    opencv-python>=4.8.0 \
    pillow>=9.5.0 \
    numpy>=1.24.0 \
    rouge-score>=0.1.2 \
    scikit-learn>=1.3.0 \
    python-dotenv>=1.0.0 \
    tqdm>=4.65.0 \
    redis>=7.0.0 \
    wandb>=0.22.0 \
    tensorboard>=2.20.0 \
    ray[default]>=2.50.0 \
    trl>=0.24.0 \
    deepspeed>=0.18.0 \
    torchmetrics>=1.8.0 \
    kornia>=0.8.0 \
    omegaconf>=2.3.0 \
    loguru>=0.7.0 \
    attrs>=25.0.0 \
    toml>=0.10.0

# Create project directories
echo "📁 Creating project structure..."
mkdir -p data/{raw_videos,processed_videos,annotations,datasets,checkpoints,results}
mkdir -p logs temp

# Set up environment variables
echo "🌍 Setting up environment variables..."
cat > .env << EOF
# Azure A100 VM Configuration
AZURE_VM=true
GPU_COUNT=1
GPU_MEMORY=80
BATCH_SIZE=4
LEARNING_RATE=2e-5
MAX_EPOCHS=3
WARMUP_STEPS=100
SAVE_STEPS=500
EVAL_STEPS=250
LOGGING_STEPS=50

# Model Configuration
MODEL_PATH=nvidia/Cosmos-Reason1-7B
LORA_PATH=./05_training/checkpoints/football_sft
DEVICE=cuda
DTYPE=bfloat16

# Training Configuration
USE_DEEPSPEED=true
USE_RAY=true
USE_WANDB=true
WANDB_PROJECT=cosmos-football-azure

# Video Processing
VIDEO_FPS=4
MAX_VIDEO_FRAMES=10
MAX_IMAGE_FRAMES=10
INPUT_CONTEXT_LENGTH=128000
EOF

# Verify installation
echo "🔍 Verifying installation..."
source cosmos-env/bin/activate
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
python -c "import torch; print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}')"

# Create startup scripts
echo "🚀 Creating startup scripts..."

# Training startup script
cat > start_training.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting Cosmos Football Training on Azure A100"
echo "=================================================================="

# Activate environment
source cosmos-env/bin/activate

# Check GPU status
echo "🔧 GPU Status:"
nvidia-smi

# Check disk space
echo "💾 Disk Space:"
df -h

# Check memory
echo "🧠 Memory Usage:"
free -h

# Start training
echo "🎯 Starting training..."
cd 05_training
python azure_training.py --config azure_training_config.toml

echo "✅ Training completed!"
EOF
chmod +x start_training.sh

# Monitoring script
cat > monitor_system.sh << 'EOF'
#!/bin/bash
echo "📊 Azure A100 System Monitoring"
echo "=================================================================="

# GPU monitoring
echo "🔧 GPU Status:"
nvidia-smi --query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv

# Memory monitoring
echo "💾 Memory Usage:"
free -h

# Disk monitoring
echo "💿 Disk Usage:"
df -h

# Network monitoring
echo "🌐 Network Status:"
ping -c 3 8.8.8.8

# Process monitoring
echo "⚙️  Running Processes:"
ps aux | grep python | head -5
EOF
chmod +x monitor_system.sh

# Final status
echo "🎉 Azure A100 Quick Start Completed Successfully!"
echo "=================================================================="
echo ""
echo "📊 Setup Summary:"
echo "  ✅ System: Ubuntu with NVIDIA drivers"
echo "  ✅ Python: Virtual environment with all dependencies"
echo "  ✅ PyTorch: CUDA-enabled with A100 support"
echo "  ✅ Cosmos: All framework dependencies installed"
echo "  ✅ Project: Directory structure created"
echo "  ✅ Configuration: Azure-optimized settings"
echo ""
echo "🚀 Next Steps:"
echo "  1. Run tests: make test"
echo "  2. Start training: make train"
echo "  3. Full validation: make validate"
echo ""
echo "📋 Available Commands:"
echo "  make help          - Show all available commands"
echo "  make status        - Check system status"
echo "  make monitor       - Monitor system performance"
echo "  make deploy        - Complete deployment pipeline"
echo "  ./start_training.sh - Start training"
echo "  ./monitor_system.sh - Monitor system"
echo ""
echo "🎯 Ready for Cosmos Football Video Analysis on Azure A100!"
echo ""
echo "💡 Quick Commands:"
echo "  source cosmos-env/bin/activate  # Activate environment"
echo "  make test                       # Run pipeline tests"
echo "  make train                      # Start training"
echo "  make validate                   # Complete validation"
