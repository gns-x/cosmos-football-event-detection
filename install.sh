# 1) System basics
sudo apt update
sudo apt install -y python3-venv git ffmpeg libgl1 libglib2.0-0

# 2) Python virtual environment
python3 -m venv cosmos
source cosmos/bin/activate
python -m pip install --upgrade pip

# 3) GPU-accelerated PyTorch (matches common Azure CUDA 12.x stacks)
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision

# 4) Model/runtime dependencies: vLLM, Transformers, Qwen-VL utils + video readers
pip install vllm "transformers>=4.51.3" "qwen-vl-utils[decord]" huggingface_hub
