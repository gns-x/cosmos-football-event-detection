#!/bin/bash

# NVIDIA NIM Cosmos Video Analysis - Full Stack Startup Script

echo "🚀 Starting NVIDIA NIM Cosmos Video Analysis Application..."

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

# Function to start NVIDIA NIM backend
start_nim_backend() {
    echo "🔧 Starting NVIDIA NIM Backend API..."
    cd backend
    
    # Check if virtual environment exists
    if [ ! -d "nim-env" ]; then
        echo "📦 Setting up NVIDIA NIM backend environment..."
        ./setup_nim.sh
    fi
    
    # Check if .env file exists and has API key
    if [ ! -f ".env" ]; then
        echo "⚠️  .env file not found! Creating from template..."
        cp .env.nim .env
        echo "❌ Please edit .env file with your NVIDIA API key before continuing!"
        echo "🔑 Get your API key from: https://build.nvidia.com/"
        exit 1
    fi
    
    # Check if NVIDIA_API_KEY is set
    if ! grep -q "NVIDIA_API_KEY=your_nvidia_api_key_here" .env; then
        if grep -q "NVIDIA_API_KEY=" .env; then
            echo "✅ NVIDIA API key found in .env file"
        else
            echo "❌ NVIDIA API key not found in .env file!"
            echo "🔑 Please add your NVIDIA API key to the .env file"
            exit 1
        fi
    else
        echo "❌ Please update .env file with your actual NVIDIA API key!"
        echo "🔑 Get your API key from: https://build.nvidia.com/"
        exit 1
    fi
    
    # Activate virtual environment and start API
    source nim-env/bin/activate
    echo "🌐 Starting NVIDIA NIM FastAPI server on http://localhost:8000"
    python main_nim.py &
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
start_nim_backend
sleep 3  # Give backend time to start
start_frontend

echo ""
echo "✅ Both services started!"
echo ""
echo "🌐 Frontend: http://localhost:5173 (or 5174)"
echo "🔧 Backend API: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo "🧠 Model: NVIDIA NIM Cosmos-Reason1-7B"
echo ""
echo "🔑 NVIDIA API Key Setup:"
echo "1. Visit: https://build.nvidia.com/"
echo "2. Sign up/Login with NVIDIA account"
echo "3. Create a new API key"
echo "4. Copy the key to backend/.env file"
echo ""
echo "Press Ctrl+C to stop both services"

# Wait for user interrupt
trap 'echo ""; echo "🛑 Stopping services..."; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0' INT

# Keep script running
wait
