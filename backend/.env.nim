# Environment variables for NVIDIA NIM Cosmos Video Analysis API

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
