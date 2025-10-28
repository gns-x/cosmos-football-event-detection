#!/bin/bash

# Cosmos Video Analysis - Full Stack Startup Script

echo "🚀 Starting Cosmos Video Analysis Application..."

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

# Function to start backend
start_backend() {
    echo "🔧 Starting Backend API..."
    cd backend
    
    # Check if virtual environment exists
    if [ ! -d "cosmos-env" ]; then
        echo "📦 Setting up backend environment..."
        ./setup.sh
    fi
    
    # Activate virtual environment and start API
    source cosmos-env/bin/activate
    echo "🌐 Starting FastAPI server on http://localhost:8000"
    python main.py &
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
start_backend
sleep 3  # Give backend time to start
start_frontend

echo ""
echo "✅ Both services started!"
echo ""
echo "🌐 Frontend: http://localhost:5173 (or 5174)"
echo "🔧 Backend API: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop both services"

# Wait for user interrupt
trap 'echo ""; echo "🛑 Stopping services..."; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0' INT

# Keep script running
wait
