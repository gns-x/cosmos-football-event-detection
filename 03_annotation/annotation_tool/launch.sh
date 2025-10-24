#!/bin/bash

# Football Video Annotation Tool Launcher
echo "🚀 Starting Football Video Annotation Tool..."

# Check if Flask is installed
if ! python3 -c "import flask" 2>/dev/null; then
    echo "📦 Installing Flask..."
    pip install flask flask-cors
fi

# Start the annotation server
echo "🌐 Starting annotation server on http://localhost:5000"
echo "📝 Open your browser and go to: http://localhost:5000"
echo "⏹️  Press Ctrl+C to stop the server"
echo ""

cd "$(dirname "$0")"
python3 app.py
