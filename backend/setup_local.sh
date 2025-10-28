#!/bin/bash

# Local Cosmos Video Analysis API Setup Script

echo "🚀 Setting up Local Cosmos Video Analysis API..."

# Check system requirements
echo "🔍 Checking system requirements..."

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo "✅ Python $python_version detected"
else
    echo "❌ Python 3.8+ required. Found: $python_version"
    exit 1
fi

# Check CUDA availability
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
else
    echo "⚠️  No NVIDIA GPU detected - will use CPU (slower)"
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv cosmos-local-env
source cosmos-local-env/bin/activate

# Upgrade pip
echo "📥 Upgrading pip..."
pip install --upgrade pip

# Install PyTorch first (with CUDA if available)
echo "🔥 Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    echo "Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "Installing PyTorch CPU version..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install other dependencies
echo "📥 Installing other dependencies..."
pip install -r requirements_local.txt

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    cp .env.local .env
    echo "✅ Environment file created"
fi

# Create necessary directories
mkdir -p logs
mkdir -p temp
mkdir -p models

echo "✅ Local setup complete!"
echo ""
echo "🔧 Next steps:"
echo "1. Run: source cosmos-local-env/bin/activate"
echo "2. Run: python main_local.py"
echo ""
echo "⚠️  Important Notes:"
echo "- First run will download the model (~14GB)"
echo "- GPU recommended for best performance"
echo "- CPU mode will be slower but functional"
echo ""
echo "🌐 API will be available at: http://localhost:8000"
echo "📚 API docs at: http://localhost:8000/docs"
echo ""
echo "💾 Model will be cached in ~/.cache/huggingface/"
