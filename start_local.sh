#!/bin/bash

# Local Cosmos Video Analysis - Full Stack Startup Script

echo "🚀 Starting Local Cosmos Video Analysis Application..."

# Check if backend directory exists
if [ ! -d "backend" ]; then
    echo "❌ Backend directory not found!"
    exit 1
fi

# Check if frontend directory exists
if [ ! -d "frontend" ]; then
    echo "❌ Frontend directory not found!"
    exit 1
fi

# Function to start local backend
start_local_backend() {
    echo "🔧 Starting Local Cosmos Backend API..."
    cd backend
    
    # Check if virtual environment exists
    if [ ! -d "cosmos-local-env" ]; then
        echo "📦 Setting up local backend environment..."
        ./setup_local.sh
    fi
    
    # Check if .env file exists
    if [ ! -f ".env" ]; then
        echo "📝 Creating .env file..."
        cp .env.local .env
    fi
    
    # Activate virtual environment and start API
    source cosmos-local-env/bin/activate
    echo "🌐 Starting Local FastAPI server on http://localhost:8000"
    echo "⚠️  First run will download the model (~14GB) - this may take a while"
    python main_local.py &
    BACKEND_PID=$!
    cd ..
}

# Function to start frontend
start_frontend() {
    echo "🎨 Starting Frontend..."
    cd frontend
    
    # Check if node_modules exists
    if [ ! -d "node_modules" ]; then
        echo "📦 Installing frontend dependencies..."
        npm install
    fi
    
    echo "🌐 Starting Vite dev server..."
    npm run dev &
    FRONTEND_PID=$!
    cd ..
}

# Start both services
start_local_backend
sleep 5  # Give backend more time to start (model loading)
start_frontend

echo ""
echo "✅ Both services started!"
echo ""
echo "🌐 Frontend: http://localhost:5173 (or 5174)"
echo "🔧 Backend API: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo "🧠 Model: NVIDIA Cosmos-Reason1-7B (Local)"
echo ""
echo "⚠️  Important Notes:"
echo "- First run downloads ~14GB model"
echo "- GPU recommended for performance"
echo "- CPU mode works but is slower"
echo "- Model cached in ~/.cache/huggingface/"
echo ""
echo "Press Ctrl+C to stop both services"

# Wait for user interrupt
trap 'echo ""; echo "🛑 Stopping services..."; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0' INT

# Keep script running
wait
