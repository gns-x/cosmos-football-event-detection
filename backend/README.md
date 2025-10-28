# Cosmos Video Analysis API

A backend API service that uses NVIDIA's Cosmos-Reason1-7B model for intelligent video analysis, specifically designed for football/sports video understanding.

## 🚀 Features

- **Cosmos-Reason1-7B Integration**: Uses NVIDIA's advanced reasoning model
- **Video Frame Extraction**: Automatically extracts key frames from uploaded videos
- **Physical AI Reasoning**: Understands spatial, temporal, and physical relationships
- **RESTful API**: Clean, documented API endpoints
- **Frontend Integration**: Ready-to-use React frontend with API service
- **Real-time Analysis**: Live video analysis with progress indicators

## 📋 Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended) or CPU
- Hugging Face account and token
- Node.js 16+ (for frontend)

## 🛠️ Setup Instructions

### 1. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Run setup script
chmod +x setup.sh
./setup.sh

# Activate virtual environment
source cosmos-env/bin/activate

# Edit environment variables
cp .env.example .env
# Add your Hugging Face token to .env file

# Install dependencies
pip install -r requirements.txt

# Start the API server
python main.py
```

The API will be available at:
- **API**: http://localhost:8000
- **Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### 2. Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The frontend will be available at:
- **Frontend**: http://localhost:5173 or http://localhost:5174

## 🔧 Configuration

### Environment Variables (.env)

```env
# Hugging Face settings
HUGGINGFACE_TOKEN=your_huggingface_token_here

# Model settings
MODEL_NAME=nvidia/Cosmos-Reason1-7B
DEVICE=auto  # auto, cuda, or cpu
MAX_TOKENS=512
TEMPERATURE=0.7

# API settings
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=True

# CORS settings
ALLOWED_ORIGINS=http://localhost:5173,http://localhost:5174,http://localhost:3000
```

## 📡 API Endpoints

### Health & Status
- `GET /` - Basic health check
- `GET /health` - Detailed health status
- `GET /model-info` - Model information

### Analysis
- `POST /analyze` - Analyze video with file upload
- `POST /analyze-text` - Text-only analysis

### Example Usage

```bash
# Health check
curl http://localhost:8000/health

# Analyze video
curl -X POST "http://localhost:8000/analyze" \
  -F "prompt=Which player scored the goal?" \
  -F "video_file=@football_match.mp4"

# Text analysis
curl -X POST "http://localhost:8000/analyze-text" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Analyze this football scenario: A player dribbles past defenders and shoots"}'
```

## 🎯 Frontend Integration

The frontend includes:

- **Video Upload**: Drag-and-drop video upload with preview
- **Real-time Analysis**: Live analysis with progress indicators
- **Backend Status**: Connection status indicator
- **Results Display**: Formatted analysis results with reasoning
- **Error Handling**: Graceful error handling and user feedback

### Key Components

- `cosmosAPI.ts` - API service for backend communication
- `App.tsx` - Main application with video player and analysis UI
- Status indicators and loading states
- Real-time backend health monitoring

## 🧠 Model Capabilities

The Cosmos-Reason1-7B model excels at:

- **Physical Reasoning**: Understanding object interactions and physics
- **Spatial Understanding**: Analyzing positions and movements
- **Temporal Reasoning**: Understanding sequences and timing
- **Common Sense**: Applying real-world knowledge to scenarios
- **Multi-modal Analysis**: Processing both visual and textual information

## 🔍 Use Cases

Perfect for:
- **Sports Analysis**: Football, basketball, soccer video analysis
- **Physical AI**: Understanding physical interactions and movements
- **Embodied Reasoning**: Robot navigation and manipulation tasks
- **Educational Content**: Explaining complex physical phenomena
- **Research Applications**: Studying human behavior and interactions

## 🚨 Troubleshooting

### Common Issues

1. **Model Loading Fails**
   - Check GPU memory availability
   - Verify Hugging Face token
   - Try CPU mode if GPU unavailable

2. **API Connection Issues**
   - Verify backend is running on port 8000
   - Check CORS settings
   - Ensure frontend and backend are on same network

3. **Video Upload Issues**
   - Check file size limits
   - Verify video format support
   - Ensure sufficient disk space for temp files

### Performance Tips

- Use GPU acceleration when available
- Limit video frame extraction for faster processing
- Implement video compression for large files
- Use quantized models for better performance

## 📄 License

This project uses the NVIDIA Open Model License for the Cosmos-Reason1-7B model.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📞 Support

For issues and questions:
- Check the troubleshooting section
- Review API documentation at `/docs`
- Open an issue on GitHub

---

**Built with ❤️ using NVIDIA Cosmos-Reason1-7B and FastAPI**
