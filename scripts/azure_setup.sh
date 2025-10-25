#!/bin/bash
# Azure A100 VM Setup Script for Cosmos Football Video Analysis
# Comprehensive setup for Azure Standard_NC24ads_A100_v4 instances

set -e

echo "☁️  Azure A100 VM Setup for Cosmos Football Analysis"
echo "=================================================================="
echo "🖥️  Azure VM Configuration:"
echo "  Instance: Standard_NC24ads_A100_v4"
echo "  Memory: 220 GB"
echo "  GPU: 1x A100 (80GB)"
echo "  Storage: 1TB SSD"
echo "  OS: Ubuntu 20.04 LTS"
echo ""

# System information
echo "📋 System Information:"
uname -a
echo ""

# Check GPU availability
echo "🔧 Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    echo "✅ NVIDIA drivers found"
else
    echo "⚠️  NVIDIA drivers not found - installing..."
    # Install NVIDIA drivers
    sudo apt-get update
    sudo apt-get install -y nvidia-driver-535
    echo "✅ NVIDIA drivers installed - reboot required"
fi
echo ""

# Update system packages
echo "📦 Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y
echo "✅ System packages updated"
echo ""

# Install system dependencies
echo "📦 Installing system dependencies..."
sudo apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    git \
    wget \
    curl \
    ffmpeg \
    build-essential \
    cmake \
    pkg-config \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran \
    yt-dlp
echo "✅ System dependencies installed"
echo ""

# Create Python virtual environment
echo "🐍 Setting up Python environment..."
python3 -m venv cosmos-env
source cosmos-env/bin/activate
pip install --upgrade pip setuptools wheel
echo "✅ Python environment created"
echo ""

# Install PyTorch with CUDA support
echo "🔥 Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
echo "✅ PyTorch installed"
echo ""

# Install core dependencies
echo "📦 Installing core dependencies..."
pip install \
    transformers>=4.30.0 \
    vllm>=0.6.0 \
    qwen_vl_utils>=0.0.1 \
    accelerate>=0.20.0 \
    peft>=0.4.0 \
    bitsandbytes>=0.39.0
echo "✅ Core dependencies installed"
echo ""

# Install video processing dependencies
echo "🎬 Installing video processing dependencies..."
pip install \
    opencv-python>=4.8.0 \
    pillow>=9.5.0 \
    numpy>=1.24.0 \
    imageio>=2.31.0 \
    imageio-ffmpeg>=0.6.0
echo "✅ Video processing dependencies installed"
echo ""

# Install evaluation dependencies
echo "📊 Installing evaluation dependencies..."
pip install \
    rouge-score>=0.1.2 \
    scikit-learn>=1.3.0 \
    torchmetrics>=1.8.0 \
    kornia>=0.8.0
echo "✅ Evaluation dependencies installed"
echo ""

# Install Cosmos RL SFT Framework dependencies
echo "🚀 Installing Cosmos RL SFT Framework dependencies..."
pip install \
    redis>=7.0.0 \
    wandb>=0.22.0 \
    tensorboard>=2.20.0 \
    tensorboardX>=2.6.0 \
    ray[default]>=2.50.0 \
    trl>=0.24.0 \
    deepspeed>=0.18.0
echo "✅ Cosmos RL SFT Framework dependencies installed"
echo ""

# Install utility dependencies
echo "🔧 Installing utility dependencies..."
pip install \
    python-dotenv>=1.0.0 \
    tqdm>=4.65.0 \
    omegaconf>=2.3.0 \
    loguru>=0.7.0 \
    attrs>=25.0.0 \
    toml>=0.10.0 \
    matplotlib>=3.10.0 \
    ipython>=9.6.0
echo "✅ Utility dependencies installed"
echo ""

# Create project directories
echo "📁 Creating project structure..."
mkdir -p data/{raw_videos,processed_videos,annotations,datasets,checkpoints,results}
mkdir -p logs
mkdir -p temp
echo "✅ Project structure created"
echo ""

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

# Azure Storage (if using)
# AZURE_STORAGE_ACCOUNT=your_account
# AZURE_STORAGE_KEY=your_key
# AZURE_CONTAINER=cosmos-football
EOF
echo "✅ Environment variables configured"
echo ""

# Verify installation
echo "🔍 Verifying installation..."
source cosmos-env/bin/activate
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
python -c "import torch; print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}')"
echo "✅ Installation verification completed"
echo ""

# Create Azure-specific configuration
echo "☁️  Creating Azure-specific configuration..."
cat > azure_config.yaml << EOF
# Azure A100 VM Configuration for Cosmos Football Analysis
azure:
  vm_type: "Standard_NC24ads_A100_v4"
  gpu_count: 1
  gpu_memory: 80
  system_memory: 220
  storage: 1000

# Performance optimizations
performance:
  use_deepspeed: true
  use_ray: true
  use_wandb: true
  mixed_precision: true
  gradient_checkpointing: true
  
# Azure-specific settings
azure_settings:
  enable_telemetry: true
  log_level: INFO
  checkpoint_frequency: 500
  backup_frequency: 1000
EOF
echo "✅ Azure configuration created"
echo ""

# Create startup script
echo "🚀 Creating startup script..."
cat > start_azure_training.sh << 'EOF'
#!/bin/bash
# Azure A100 Training Startup Script

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
python simple_football_sft.py --config football_sft_config.toml

echo "✅ Training completed!"
EOF
chmod +x start_azure_training.sh
echo "✅ Startup script created"
echo ""

# Create monitoring script
echo "📊 Creating monitoring script..."
cat > monitor_azure.sh << 'EOF'
#!/bin/bash
# Azure A100 Monitoring Script

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
ps aux | grep python | head -10
EOF
chmod +x monitor_azure.sh
echo "✅ Monitoring script created"
echo ""

# Final status
echo "🎉 Azure A100 VM Setup Completed Successfully!"
echo "=================================================================="
echo ""
echo "📊 Setup Summary:"
echo "  ✅ System: Ubuntu 20.04 LTS with NVIDIA drivers"
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
echo ""
echo "🎯 Ready for Cosmos Football Video Analysis on Azure A100!"
