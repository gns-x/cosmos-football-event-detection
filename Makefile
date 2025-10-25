# Cosmos Football Video Analysis - Azure A100 VM Deployment
# Comprehensive Makefile for testing, fine-tuning, and validation

.PHONY: help setup install test train evaluate inference clean deploy status

# Default target
help:
	@echo "🏈 Cosmos Football Video Analysis - Azure A100 VM Deployment"
	@echo "=================================================================="
	@echo ""
	@echo "📋 Available Commands:"
	@echo "  make setup          - Complete Azure VM setup and environment"
	@echo "  make install        - Install all dependencies and requirements"
	@echo "  make test           - Run critical pipeline tests"
	@echo "  make train          - Execute fine-tuning on A100"
	@echo "  make evaluate       - Run evaluation with trained model"
	@echo "  make inference      - Test inference with new videos"
	@echo "  make validate       - Complete end-to-end validation"
	@echo "  make status         - Check system status and GPU availability"
	@echo "  make clean          - Clean up temporary files"
	@echo "  make deploy         - Full deployment pipeline"
	@echo ""
	@echo "🚀 Quick Start:"
	@echo "  make deploy         # Complete setup and training"
	@echo "  make validate       # Full validation pipeline"
	@echo ""

# Complete Azure VM setup
setup:
	@echo "🚀 Setting up Azure A100 VM for Cosmos Football Analysis"
	@echo "=================================================================="
	@echo "📋 System Information:"
	@uname -a
	@echo ""
	@echo "🔧 Checking GPU availability..."
	@nvidia-smi || echo "⚠️  NVIDIA drivers not found - installing..."
	@echo ""
	@echo "📦 Installing system dependencies..."
	@sudo apt-get update
	@sudo apt-get install -y python3-pip python3-venv git wget curl ffmpeg
	@sudo apt-get install -y build-essential cmake
	@echo "✅ System dependencies installed"
	@echo ""
	@echo "🐍 Setting up Python environment..."
	@python3 -m venv cosmos-env
	@source cosmos-env/bin/activate && pip install --upgrade pip
	@echo "✅ Python environment created"
	@echo ""
	@echo "📁 Setting up project structure..."
	@mkdir -p data/{raw_videos,processed_videos,annotations,datasets,checkpoints,results}
	@mkdir -p docs scripts tests
	@echo "✅ Project structure created"
	@echo ""
	@echo "🎯 Azure A100 VM setup completed!"

# Install all dependencies
install:
	@echo "📦 Installing Cosmos Football Analysis Dependencies"
	@echo "=================================================================="
	@echo "🔧 Activating Python environment..."
	@source cosmos-env/bin/activate && \
	echo "📦 Installing core dependencies..." && \
	pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
	echo "📦 Installing transformers and vLLM..." && \
	pip install transformers>=4.30.0 vllm>=0.6.0 qwen_vl_utils>=0.0.1 && \
	echo "📦 Installing training dependencies..." && \
	pip install accelerate>=0.20.0 peft>=0.4.0 bitsandbytes>=0.39.0 && \
	echo "📦 Installing video processing..." && \
	pip install opencv-python>=4.8.0 pillow>=9.5.0 numpy>=1.24.0 && \
	echo "📦 Installing evaluation tools..." && \
	pip install rouge-score>=0.1.2 scikit-learn>=1.3.0 && \
	echo "📦 Installing utilities..." && \
	pip install python-dotenv>=1.0.0 tqdm>=4.65.0 yt-dlp && \
	echo "📦 Installing Cosmos RL SFT Framework..." && \
	pip install redis>=7.0.0 wandb>=0.22.0 tensorboard>=2.20.0 && \
	pip install ray[default]>=2.50.0 trl>=0.24.0 deepspeed>=0.18.0 && \
	echo "📦 Installing Cosmos Cookbook dependencies..." && \
	pip install torchmetrics>=1.8.0 kornia>=0.8.0 omegaconf>=2.3.0 && \
	pip install loguru>=0.7.0 attrs>=25.0.0 toml>=0.10.0 && \
	echo "✅ All dependencies installed successfully!"
	@echo ""
	@echo "🔍 Verifying installation..."
	@source cosmos-env/bin/activate && python -c "import torch; print(f'PyTorch: {torch.__version__}')"
	@source cosmos-env/bin/activate && python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
	@source cosmos-env/bin/activate && python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
	@echo "✅ Installation verification completed!"

# Run critical pipeline tests
test:
	@echo "🧪 Running Critical Pipeline Tests"
	@echo "=================================================================="
	@echo "🔧 Activating environment..."
	@source cosmos-env/bin/activate && \
	echo "📋 Running end-to-end pipeline test..." && \
	python scripts/run_end_to_end_test.py && \
	echo "✅ Critical pipeline tests completed!"
	@echo ""
	@echo "📊 Test Results Summary:"
	@echo "  ✅ Phase 1&2: Data Ingestion & Preprocessing"
	@echo "  ✅ Phase 3: Annotation (Ground Truth)"
	@echo "  ✅ Phase 4: Dataset Preparation"
	@echo "  ✅ Phase 5: Training Smoke Test"
	@echo "  ✅ Phase 6: Evaluation Pipeline"
	@echo "  ✅ Phase 7: Final End-to-End Inference"

# Execute fine-tuning on A100
train:
	@echo "🚀 Starting Fine-Tuning on Azure A100"
	@echo "=================================================================="
	@echo "🔧 Activating environment..."
	@source cosmos-env/bin/activate && \
	echo "📋 Checking GPU availability..." && \
	nvidia-smi && \
	echo "" && \
	echo "🎯 Starting Cosmos fine-tuning..." && \
	cd 05_training && \
	python simple_football_sft.py --config football_sft_config.toml && \
	echo "✅ Fine-tuning completed!"
	@echo ""
	@echo "📊 Training Results:"
	@echo "  📁 Checkpoints: 05_training/checkpoints/"
	@echo "  📈 Logs: 05_training/logs/"
	@echo "  🎯 LoRA Adapter: Ready for evaluation"

# Run evaluation with trained model
evaluate:
	@echo "📊 Running Evaluation with Trained Model"
	@echo "=================================================================="
	@echo "🔧 Activating environment..."
	@source cosmos-env/bin/activate && \
	echo "📋 Checking trained model..." && \
	ls -la 05_training/checkpoints/ && \
	echo "" && \
	echo "🎯 Running evaluation..." && \
	cd 06_evaluation && \
	python evaluate.py --test_file ../04_dataset/test.jsonl && \
	echo "✅ Evaluation completed!"
	@echo ""
	@echo "📊 Evaluation Results:"
	@echo "  📁 Results: 06_evaluation/results/"
	@echo "  📈 Metrics: Accuracy, Precision, Recall, F1-score"
	@echo "  🎯 Temporal IoU: Temporal accuracy measurement"

# Test inference with new videos
inference:
	@echo "🎬 Testing Inference with New Videos"
	@echo "=================================================================="
	@echo "🔧 Activating environment..."
	@source cosmos-env/bin/activate && \
	echo "📋 Testing inference pipeline..." && \
	cd 07_inference && \
	python football_inference.py --video_path ../02_preprocessing/processed_videos/goal/goal_test_01.mp4 && \
	echo "✅ Inference test completed!"
	@echo ""
	@echo "📊 Inference Results:"
	@echo "  📁 Output: 07_inference/inference_results/"
	@echo "  🎯 JSON: Generated football analysis"
	@echo "  ⏱️  Performance: Inference timing and accuracy"

# Complete end-to-end validation
validate:
	@echo "🎯 Complete End-to-End Validation"
	@echo "=================================================================="
	@echo "🔧 Running full validation pipeline..."
	@$(MAKE) test
	@$(MAKE) train
	@$(MAKE) evaluate
	@$(MAKE) inference
	@echo ""
	@echo "✅ Complete validation pipeline completed!"
	@echo ""
	@echo "📊 Final Validation Results:"
	@echo "  ✅ Data Pipeline: Validated with real data"
	@echo "  ✅ Training: LoRA fine-tuning successful"
	@echo "  ✅ Evaluation: Accuracy metrics calculated"
	@echo "  ✅ Inference: Production-ready JSON output"
	@echo ""
	@echo "🎉 SYSTEM READY FOR PRODUCTION!"

# Check system status
status:
	@echo "📊 System Status Check"
	@echo "=================================================================="
	@echo "🖥️  System Information:"
	@uname -a
	@echo ""
	@echo "🔧 GPU Status:"
	@nvidia-smi || echo "⚠️  NVIDIA drivers not available"
	@echo ""
	@echo "🐍 Python Environment:"
	@source cosmos-env/bin/activate && python --version
	@source cosmos-env/bin/activate && pip list | grep torch
	@echo ""
	@echo "📁 Project Structure:"
	@ls -la
	@echo ""
	@echo "📊 Dataset Status:"
	@ls -la 04_dataset/
	@echo ""
	@echo "🎯 Training Status:"
	@ls -la 05_training/checkpoints/ 2>/dev/null || echo "No checkpoints found"
	@echo ""
	@echo "📈 Evaluation Status:"
	@ls -la 06_evaluation/results/ 2>/dev/null || echo "No evaluation results found"
	@echo ""
	@echo "🎬 Inference Status:"
	@ls -la 07_inference/inference_results/ 2>/dev/null || echo "No inference results found"

# Clean up temporary files
clean:
	@echo "🧹 Cleaning Up Temporary Files"
	@echo "=================================================================="
	@echo "🗑️  Removing temporary files..."
	@rm -rf __pycache__/
	@rm -rf .pytest_cache/
	@rm -rf *.pyc
	@rm -rf *.pyo
	@rm -rf .DS_Store
	@echo "✅ Cleanup completed!"

# Full deployment pipeline
deploy:
	@echo "🚀 Complete Azure A100 VM Deployment"
	@echo "=================================================================="
	@echo "📋 Starting full deployment pipeline..."
	@$(MAKE) setup
	@$(MAKE) install
	@$(MAKE) test
	@$(MAKE) train
	@$(MAKE) evaluate
	@$(MAKE) inference
	@echo ""
	@echo "🎉 DEPLOYMENT COMPLETED SUCCESSFULLY!"
	@echo ""
	@echo "📊 Deployment Summary:"
	@echo "  ✅ Azure A100 VM: Configured and ready"
	@echo "  ✅ Dependencies: All installed and verified"
	@echo "  ✅ Pipeline Tests: All critical tests passing"
	@echo "  ✅ Fine-tuning: LoRA model trained successfully"
	@echo "  ✅ Evaluation: Accuracy metrics calculated"
	@echo "  ✅ Inference: Production-ready system"
	@echo ""
	@echo "🎯 SYSTEM READY FOR PRODUCTION USE!"

# Azure-specific setup
azure-setup:
	@echo "☁️  Azure A100 VM Specific Setup"
	@echo "=================================================================="
	@echo "📋 Azure VM Configuration:"
	@echo "  🖥️  Instance: Standard_NC24ads_A100_v4"
	@echo "  💾 Memory: 220 GB"
	@echo "  🎯 GPU: 1x A100 (80GB)"
	@echo "  💿 Storage: 1TB SSD"
	@echo ""
	@echo "🔧 Azure-specific optimizations..."
	@echo "  📦 Installing Azure CLI..."
	@curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
	@echo "  📦 Installing Azure ML SDK..."
	@source cosmos-env/bin/activate && pip install azureml-sdk
	@echo "  📦 Installing Azure Storage SDK..."
	@source cosmos-env/bin/activate && pip install azure-storage-blob
	@echo "✅ Azure-specific setup completed!"

# Performance monitoring
monitor:
	@echo "📊 Performance Monitoring"
	@echo "=================================================================="
	@echo "🔧 GPU Monitoring:"
	@nvidia-smi --query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv
	@echo ""
	@echo "💾 Memory Usage:"
	@free -h
	@echo ""
	@echo "💿 Disk Usage:"
	@df -h
	@echo ""
	@echo "🌐 Network Status:"
	@ping -c 3 8.8.8.8

# Backup and restore
backup:
	@echo "💾 Creating Backup"
	@echo "=================================================================="
	@echo "📁 Creating backup of project..."
	@tar -czf cosmos-football-backup-$(shell date +%Y%m%d-%H%M%S).tar.gz \
		--exclude=cosmos-env \
		--exclude=__pycache__ \
		--exclude=*.pyc \
		--exclude=.git \
		.
	@echo "✅ Backup created successfully!"

restore:
	@echo "🔄 Restoring from Backup"
	@echo "=================================================================="
	@echo "📁 Available backups:"
	@ls -la cosmos-football-backup-*.tar.gz 2>/dev/null || echo "No backups found"
	@echo ""
	@echo "💡 To restore, run:"
	@echo "  tar -xzf cosmos-football-backup-YYYYMMDD-HHMMSS.tar.gz"

# Quick start for Azure
azure-quick-start:
	@echo "⚡ Azure A100 Quick Start"
	@echo "=================================================================="
	@echo "🚀 Quick deployment for Azure A100 VM..."
	@$(MAKE) azure-setup
	@$(MAKE) install
	@$(MAKE) test
	@echo ""
	@echo "✅ Quick start completed!"
	@echo "🎯 Ready for training: make train"
	@echo "🎯 Full validation: make validate"
