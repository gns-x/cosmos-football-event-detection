# Local Cosmos Video Analysis API

A complete local deployment of NVIDIA's Cosmos-Reason1-7B model using Hugging Face for intelligent video analysis. **100% FREE and runs entirely on your machine!**

## 🎯 Why Local Deployment?

- **✅ 100% FREE**: No API costs, no usage limits
- **🔒 Complete Privacy**: Your data never leaves your machine
- **⚡ No Internet Required**: Works offline after initial download
- **🎛️ Full Control**: Customize model behavior and parameters
- **🚀 No Rate Limits**: Process as many videos as you want
- **💰 One-time Setup**: Download once, use forever

## 🚀 Quick Start

### **Option 1: Automated Setup (Recommended)**
```bash
# One command to start everything
./start_local.sh
```

### **Option 2: Manual Setup**
```bash
# Backend setup
cd backend
./setup_local.sh
source cosmos-local-env/bin/activate
python main_local.py

# Frontend setup (in new terminal)
cd frontend
npm install
npm run dev
```

## 📋 System Requirements

### **Minimum Requirements**
- **RAM**: 16GB (32GB recommended)
- **Storage**: 20GB free space
- **Python**: 3.8+
- **OS**: Linux, macOS, or Windows

### **Recommended Setup**
- **GPU**: NVIDIA GPU with 8GB+ VRAM
- **RAM**: 32GB+
- **Storage**: SSD with 50GB+ free space
- **CUDA**: 11.8+ (for GPU acceleration)

## 🔧 Configuration

### Environment Variables (.env)
```env
# Model settings
MODEL_NAME=nvidia/Cosmos-Reason1-7B
DEVICE=auto  # auto, cuda, or cpu
USE_QUANTIZATION=true  # 4-bit quantization for GPU
MAX_TOKENS=512
TEMPERATURE=0.7

# API settings
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=True

# Memory optimization
CLEAR_CACHE_ON_REQUEST=false
MAX_MEMORY_USAGE=0.8
```

## 📡 API Endpoints

### Health & Status
- `GET /` - Basic health check
- `GET /health` - Detailed health status with memory info
- `GET /model-info` - Model information and memory usage
- `POST /clear-cache` - Clear GPU memory cache

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

# Clear GPU cache
curl -X POST "http://localhost:8000/clear-cache"
```

## 🧠 Model Capabilities

The local Cosmos-Reason1-7B model excels at:

- **Physical Reasoning**: Understanding object interactions and physics
- **Spatial Understanding**: Analyzing positions and movements
- **Temporal Reasoning**: Understanding sequences and timing
- **Common Sense**: Applying real-world knowledge to scenarios
- **Multi-modal Analysis**: Processing both visual and textual information

## ⚡ Performance Optimization

### **GPU Optimization**
- **4-bit Quantization**: Reduces memory usage by ~75%
- **Mixed Precision**: Uses FP16 for faster inference
- **Memory Management**: Automatic GPU memory cleanup
- **Batch Processing**: Optimized for multiple requests

### **CPU Optimization**
- **Model Quantization**: Smaller model size for CPU
- **Memory Mapping**: Efficient memory usage
- **Threading**: Multi-threaded inference
- **Caching**: Intelligent response caching

### **Memory Management**
```python
# Automatic memory cleanup
torch.cuda.empty_cache()  # Clear GPU cache
gc.collect()              # Garbage collection
```

## 🔍 Use Cases

Perfect for:
- **Sports Analysis**: Football, basketball, soccer video analysis
- **Physical AI**: Understanding physical interactions and movements
- **Educational Content**: Explaining complex physical phenomena
- **Research Applications**: Studying human behavior and interactions
- **Offline Analysis**: No internet required after setup

## 🚨 Troubleshooting

### **Common Issues**

#### **1. Out of Memory (OOM)**
```bash
# Solutions:
# - Enable quantization: USE_QUANTIZATION=true
# - Reduce max_tokens: MAX_TOKENS=256
# - Use CPU: DEVICE=cpu
# - Clear cache: POST /clear-cache
```

#### **2. Slow Performance**
```bash
# Solutions:
# - Use GPU: DEVICE=cuda
# - Enable quantization: USE_QUANTIZATION=true
# - Reduce video frames: MAX_FRAMES=5
# - Use SSD storage
```

#### **3. Model Download Issues**
```bash
# Solutions:
# - Check internet connection
# - Clear Hugging Face cache: rm -rf ~/.cache/huggingface/
# - Use VPN if region blocked
# - Manual download from Hugging Face
```

### **Performance Tips**

1. **First Run**: Model download takes time (~14GB)
2. **GPU Memory**: Monitor with `nvidia-smi`
3. **Quantization**: Use 4-bit for memory efficiency
4. **Batch Size**: Process multiple videos together
5. **Caching**: Cache responses for repeated queries

## 📊 Memory Usage

### **GPU Memory Requirements**
- **Full Precision**: ~14GB VRAM
- **4-bit Quantized**: ~4GB VRAM
- **8-bit Quantized**: ~7GB VRAM

### **RAM Requirements**
- **Model Loading**: ~16GB RAM
- **Inference**: ~8GB RAM
- **Video Processing**: ~4GB RAM

## 🔄 Model Management

### **Download Location**
```bash
# Default cache location
~/.cache/huggingface/hub/models--nvidia--Cosmos-Reason1-7B/
```

### **Manual Download**
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Download model manually
tokenizer = AutoTokenizer.from_pretrained("nvidia/Cosmos-Reason1-7B")
model = AutoModelForCausalLM.from_pretrained("nvidia/Cosmos-Reason1-7B")
```

### **Model Updates**
```bash
# Update to latest version
pip install --upgrade transformers
# Restart API to reload model
```

## 🆚 Comparison: Local vs Cloud

| Feature | Local Deployment | Cloud API |
|---------|------------------|-----------|
| **Cost** | ⭐⭐⭐⭐⭐ FREE | ⭐⭐ Pay-per-use |
| **Privacy** | ⭐⭐⭐⭐⭐ Complete | ⭐⭐ Shared |
| **Speed** | ⭐⭐⭐⭐ Fast | ⭐⭐⭐⭐⭐ Very Fast |
| **Setup** | ⭐⭐⭐ Complex | ⭐⭐⭐⭐⭐ Simple |
| **Offline** | ⭐⭐⭐⭐⭐ Yes | ❌ No |
| **Customization** | ⭐⭐⭐⭐⭐ Full | ⭐⭐ Limited |

## 📞 Support

### **Local Deployment Issues**
- Check system requirements
- Monitor memory usage
- Review error logs
- Clear model cache if needed

### **Performance Issues**
- Enable GPU acceleration
- Use quantization
- Monitor resource usage
- Optimize video processing

## 🎉 Benefits Summary

✅ **100% FREE** - No ongoing costs  
✅ **Complete Privacy** - Data stays local  
✅ **No Internet Required** - Works offline  
✅ **No Rate Limits** - Process unlimited videos  
✅ **Full Control** - Customize everything  
✅ **One-time Setup** - Download once, use forever  

---

**Built with ❤️ using NVIDIA Cosmos-Reason1-7B and Hugging Face**

**100% Local, 100% Free! 🎉**
