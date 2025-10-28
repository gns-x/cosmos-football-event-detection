# Azure A100 VM Setup Guide - Cosmos Video Analysis

Complete setup guide for running Cosmos-Reason1-7B locally on Azure A100 VM.

## 🚀 Quick Start (One Command)

```bash
# Clone the repository
git clone https://github.com/gns-x/cosmos-football-event-detection.git
cd cosmos-football-event-detection

# Complete setup with one command
make init
```

## 📋 What `make init` Does

The `make init` command sets up everything automatically:

1. **🔧 System Setup**: Installs dependencies and creates Python environment
2. **📦 Dependencies**: Installs PyTorch, transformers, FastAPI, and all requirements
3. **🧠 Cosmos Model**: Downloads Cosmos-Reason1-7B (~14GB) from Hugging Face
4. **🌐 Backend API**: Sets up FastAPI server with local model integration
5. **🎨 Frontend**: Installs React frontend with video upload capabilities
6. **🧪 Testing**: Runs CLI tests to verify the model works correctly

## 🎯 Manual Steps (If Needed)

### 1. Clone Repository
```bash
git clone https://github.com/gns-x/cosmos-football-event-detection.git
cd cosmos-football-event-detection
```

### 2. Run Complete Setup
```bash
make init
```

### 3. Test the Model
```bash
make test-cosmos
```

### 4. Start the System
```bash
# Terminal 1: Start Backend API
cd backend
source cosmos-local-env/bin/activate
python main_local.py

# Terminal 2: Start Frontend
cd frontend
npm run dev
```

## 🌐 Access Points

- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 🧪 Testing Commands

### Test Cosmos Model Only
```bash
make test-cosmos
```

### Test with CLI Script
```bash
python test_cosmos_cli.py
```

### Test API Endpoints
```bash
# Health check
curl http://localhost:8000/health

# Test analysis
curl -X POST "http://localhost:8000/analyze-text" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Analyze this football scenario: A player scores a goal"}'
```

## 📊 System Requirements

### Minimum Requirements
- **RAM**: 32GB (recommended for A100)
- **Storage**: 50GB free space
- **GPU**: NVIDIA A100 (80GB VRAM)
- **CUDA**: 11.8+

### Expected Performance
- **Model Download**: ~10-30 minutes (14GB)
- **Model Loading**: ~2-5 minutes
- **Inference Speed**: ~1-3 seconds per query
- **Memory Usage**: ~40-60GB VRAM

## 🔧 Troubleshooting

### Common Issues

#### 1. Out of Memory
```bash
# Check GPU memory
nvidia-smi

# Clear GPU cache
curl -X POST "http://localhost:8000/clear-cache"
```

#### 2. Model Download Issues
```bash
# Check internet connection
ping huggingface.co

# Clear Hugging Face cache
rm -rf ~/.cache/huggingface/

# Retry download
make test-cosmos
```

#### 3. Dependencies Issues
```bash
# Fix dependencies
make fix-deps

# Reinstall everything
rm -rf cosmos-env/
make init
```

### Performance Optimization

#### GPU Memory Optimization
```bash
# Enable quantization (in .env)
USE_QUANTIZATION=true
MAX_TOKENS=256
```

#### CPU Fallback
```bash
# Force CPU mode (in .env)
DEVICE=cpu
```

## 📁 Project Structure

```
cosmos-football-event-detection/
├── backend/
│   ├── main_local.py          # Local Cosmos API
│   ├── requirements_local.txt  # Local dependencies
│   ├── setup_local.sh         # Backend setup script
│   └── cosmos-local-env/      # Backend virtual environment
├── frontend/
│   ├── src/
│   │   ├── App.tsx           # Main React app
│   │   └── services/
│   │       └── cosmosAPI.ts  # API service
│   └── node_modules/         # Frontend dependencies
├── test_cosmos_cli.py        # CLI test script
├── Makefile                  # Main setup commands
└── README.md                 # This file
```

## 🎯 Key Features

### ✅ 100% Free Local Deployment
- No API costs
- No usage limits
- No rate limits
- Complete privacy

### ✅ Azure A100 Optimized
- GPU acceleration
- 4-bit quantization
- Memory optimization
- High performance

### ✅ Complete System
- Backend API with FastAPI
- Frontend with React
- Video upload and analysis
- Real-time processing

### ✅ Easy Testing
- CLI test script
- API endpoint testing
- Health monitoring
- Performance metrics

## 🚀 Production Deployment

### Start Backend Service
```bash
cd backend
source cosmos-local-env/bin/activate
python main_local.py
```

### Start Frontend Service
```bash
cd frontend
npm run dev
```

### Monitor System
```bash
# Check GPU usage
nvidia-smi

# Check API health
curl http://localhost:8000/health

# Check model info
curl http://localhost:8000/model-info
```

## 📞 Support

### Quick Commands
```bash
make help          # Show all available commands
make status        # Check system status
make clean         # Clean temporary files
make test-cosmos   # Test Cosmos model
```

### Logs and Debugging
```bash
# Backend logs
cd backend && tail -f logs/*.log

# Frontend logs
cd frontend && npm run dev

# System monitoring
nvidia-smi -l 1
```

---

**🎉 Ready to analyze football videos with Cosmos-Reason1-7B!**

**One command setup: `make init`**
