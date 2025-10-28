# NVIDIA NIM Cosmos Video Analysis API

A backend API service that uses NVIDIA's NIM (NVIDIA Inference Microservices) platform to access the Cosmos-Reason1-7B model for intelligent video analysis. **No Hugging Face required!**

## 🎯 Why NVIDIA NIM?

- **✅ No Hugging Face Required**: Direct access to NVIDIA's models
- **🚀 Official NVIDIA Platform**: Enterprise-grade inference service
- **⚡ High Performance**: Optimized for NVIDIA hardware
- **🔒 Secure**: NVIDIA-managed infrastructure
- **💰 Cost Effective**: Pay-per-use pricing model
- **🌐 Global Availability**: Worldwide API endpoints

## 🚀 Quick Start

### 1. Get NVIDIA API Key

1. Visit [NVIDIA Build](https://build.nvidia.com/)
2. Sign up/Login with your NVIDIA account
3. Navigate to API Keys section
4. Create a new API key
5. Copy the key for later use

### 2. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Run NVIDIA NIM setup
chmod +x setup_nim.sh
./setup_nim.sh

# Activate virtual environment
source nim-env/bin/activate

# Edit environment variables
cp .env.nim .env
# Add your NVIDIA API key to .env file

# Start the API server
python main_nim.py
```

### 3. Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

## 🔧 Configuration

### Environment Variables (.env)

```env
# NVIDIA NIM settings
NVIDIA_API_KEY=your_nvidia_api_key_here
NIM_BASE_URL=https://api.nim.nvidia.com/v1
NIM_MODEL_NAME=cosmos-reason1-7b

# API settings
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=True

# CORS settings
ALLOWED_ORIGINS=http://localhost:5173,http://localhost:5174,http://localhost:3000

# Video processing settings
MAX_FRAMES=10
FRAME_SKIP=30
JPEG_QUALITY=85

# Model settings
MAX_TOKENS=512
TEMPERATURE=0.7
TIMEOUT_SECONDS=60
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

## 🧠 NVIDIA NIM Advantages

### **Performance Benefits**
- **Optimized Inference**: NVIDIA-optimized model serving
- **Low Latency**: Sub-second response times
- **High Throughput**: Handles multiple concurrent requests
- **Auto-scaling**: Automatically scales based on demand

### **Reliability Features**
- **99.9% Uptime**: Enterprise-grade reliability
- **Global CDN**: Fast response times worldwide
- **Automatic Failover**: Built-in redundancy
- **Monitoring**: Real-time performance metrics

### **Security & Compliance**
- **Enterprise Security**: SOC 2 compliant
- **Data Privacy**: Your data stays secure
- **API Authentication**: Secure token-based access
- **Rate Limiting**: Built-in protection against abuse

## 🎯 Model Capabilities

The Cosmos-Reason1-7B model via NVIDIA NIM excels at:

- **Physical Reasoning**: Understanding object interactions and physics
- **Spatial Understanding**: Analyzing positions and movements
- **Temporal Reasoning**: Understanding sequences and timing
- **Common Sense**: Applying real-world knowledge to scenarios
- **Multi-modal Analysis**: Processing both visual and textual information

## 💰 Pricing

NVIDIA NIM offers competitive pricing:

- **Pay-per-use**: Only pay for what you use
- **No upfront costs**: No infrastructure setup required
- **Transparent pricing**: Clear per-token pricing model
- **Free tier available**: Get started with free credits

## 🔍 Use Cases

Perfect for:
- **Sports Analysis**: Football, basketball, soccer video analysis
- **Physical AI**: Understanding physical interactions and movements
- **Embodied Reasoning**: Robot navigation and manipulation tasks
- **Educational Content**: Explaining complex physical phenomena
- **Research Applications**: Studying human behavior and interactions

## 🚨 Troubleshooting

### Common Issues

1. **API Key Issues**
   - Verify your NVIDIA API key is correct
   - Check if the key has proper permissions
   - Ensure the key is not expired

2. **Rate Limiting**
   - NVIDIA NIM has rate limits
   - Implement exponential backoff
   - Consider upgrading your plan for higher limits

3. **Model Availability**
   - Check if cosmos-reason1-7b is available in your region
   - Verify the model name is correct
   - Check NVIDIA NIM status page

### Performance Tips

- **Batch Requests**: Group multiple requests when possible
- **Optimize Prompts**: Use clear, concise prompts
- **Image Compression**: Compress video frames for faster processing
- **Caching**: Cache results for repeated queries

## 📊 Monitoring & Analytics

NVIDIA NIM provides:

- **Usage Analytics**: Track API usage and costs
- **Performance Metrics**: Monitor response times and throughput
- **Error Tracking**: Detailed error logs and debugging
- **Custom Dashboards**: Build monitoring dashboards

## 🔄 Migration from Hugging Face

If you're migrating from Hugging Face:

1. **Replace Model Loading**: Remove HF transformers code
2. **Update API Calls**: Use NVIDIA NIM endpoints
3. **Modify Authentication**: Switch to NVIDIA API keys
4. **Test Thoroughly**: Verify all functionality works

## 📞 Support

### NVIDIA NIM Support
- **Documentation**: https://docs.nvidia.com/nim/
- **Community**: NVIDIA Developer Forums
- **Enterprise Support**: Available with paid plans

### This Project Support
- Check the troubleshooting section
- Review API documentation at `/docs`
- Open an issue on GitHub

## 🆚 Comparison: NVIDIA NIM vs Hugging Face

| Feature | NVIDIA NIM | Hugging Face |
|---------|------------|--------------|
| **Setup Complexity** | ⭐⭐⭐⭐⭐ Simple | ⭐⭐⭐ Complex |
| **Performance** | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐⭐ Good |
| **Reliability** | ⭐⭐⭐⭐⭐ Enterprise | ⭐⭐⭐⭐ Good |
| **Cost** | ⭐⭐⭐⭐ Pay-per-use | ⭐⭐⭐ Free/Paid |
| **Model Access** | ⭐⭐⭐⭐⭐ Direct | ⭐⭐⭐⭐ Via HF |
| **No External Dependencies** | ✅ Yes | ❌ No |

---

**Built with ❤️ using NVIDIA NIM and Cosmos-Reason1-7B**

**No Hugging Face required! 🎉**
