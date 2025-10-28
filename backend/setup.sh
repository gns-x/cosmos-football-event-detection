#!/bin/bash

# Cosmos Video Analysis API Setup Script

echo "🚀 Setting up Cosmos Video Analysis API..."

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv cosmos-env
source cosmos-env/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your Hugging Face token"
fi

# Create logs directory
mkdir -p logs

# Create temp directory for video processing
mkdir -p temp

echo "✅ Setup complete!"
echo ""
echo "🔧 Next steps:"
echo "1. Edit .env file with your Hugging Face token"
echo "2. Run: source cosmos-env/bin/activate"
echo "3. Run: python main.py"
echo ""
echo "🌐 API will be available at: http://localhost:8000"
echo "📚 API docs at: http://localhost:8000/docs"
