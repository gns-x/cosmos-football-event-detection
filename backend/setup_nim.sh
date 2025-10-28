#!/bin/bash

# NVIDIA NIM Cosmos Video Analysis API Setup Script

echo "🚀 Setting up NVIDIA NIM Cosmos Video Analysis API..."

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv nim-env
source nim-env/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements_nim.txt

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    cp .env.nim .env
    echo "⚠️  Please edit .env file with your NVIDIA API key"
fi

# Create logs directory
mkdir -p logs

# Create temp directory for video processing
mkdir -p temp

echo "✅ NVIDIA NIM setup complete!"
echo ""
echo "🔧 Next steps:"
echo "1. Get NVIDIA API key from: https://build.nvidia.com/"
echo "2. Edit .env file with your NVIDIA API key"
echo "3. Run: source nim-env/bin/activate"
echo "4. Run: python main_nim.py"
echo ""
echo "🌐 API will be available at: http://localhost:8000"
echo "📚 API docs at: http://localhost:8000/docs"
echo ""
echo "🔑 NVIDIA API Key Setup:"
echo "1. Visit: https://build.nvidia.com/"
echo "2. Sign up/Login with NVIDIA account"
echo "3. Create a new API key"
echo "4. Copy the key to your .env file"
