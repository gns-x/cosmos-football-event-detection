#!/bin/bash

# NVIDIA NIM Setup Script
# Sets up the backend to use NVIDIA NIM API

echo "🚀 Setting up NVIDIA NIM Backend..."

# Check if .env file exists
if [ ! -f "backend/.env" ]; then
    echo "📝 Creating .env file from template..."
    cp backend/.env.nim backend/.env
    echo "⚠️  Please edit backend/.env and add your NVIDIA API key!"
    echo "   Get your API key from: https://build.nvidia.com/"
    echo ""
    echo "   Required environment variables:"
    echo "   - NVIDIA_API_KEY=your_api_key_here"
    echo "   - NIM_BASE_URL=https://integrate.api.nvidia.com/v1"
    echo "   - NIM_MODEL_NAME=nvidia/cosmos-reason1-7b"
    echo ""
    read -p "Press Enter after you've added your API key to backend/.env..."
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
cd backend
pip install -r requirements_nim.txt

# Verify installation
echo "✅ Dependencies installed successfully!"

echo ""
echo "🎯 Setup complete! To start the NVIDIA NIM backend:"
echo ""
echo "   cd backend"
echo "   python main_nim.py"
echo ""
echo "   Or with uvicorn:"
echo "   uvicorn main_nim:app --host 0.0.0.0 --port 8000 --reload"
echo ""
echo "🔑 Make sure your NVIDIA_API_KEY is set in backend/.env"
echo "🌐 Frontend will connect to http://localhost:8000"
echo ""
echo "📚 NVIDIA NIM Documentation: https://docs.nvidia.com/nim/"
