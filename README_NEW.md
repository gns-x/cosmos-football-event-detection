# Cosmos Football Video Analysis Project

This project fine-tunes NVIDIA's Cosmos model for football video analysis and description generation using Azure A100 VMs.

## 🎯 **Project Overview**

The Cosmos Football Video Analysis system provides end-to-end football video analysis using NVIDIA's Cosmos-Reason1-7B model with LoRA fine-tuning. The system can identify and describe football events including goals, cards, shots, and other significant actions.

## 📁 **Project Structure**

```
cosmos-football-event-detection/
├── 01_data_collection/          # Video download and collection
├── 02_preprocessing/           # Video preprocessing to 4fps
├── 03_annotation/              # Manual annotation system
├── 04_dataset/                 # Dataset preparation and formatting
├── 05_training/                # Model fine-tuning with LoRA
├── 06_evaluation/              # Model evaluation and metrics
├── 07_inference/               # Production inference system
├── docs/                       # 📚 All documentation files
├── scripts/                    # 🔧 All executable scripts
├── tests/                      # 🧪 All test files and results
├── Makefile                    # Deployment automation
├── environment.yml             # Conda environment
└── README.md                   # This file
```

### 📚 **Documentation (`docs/`)**
- `AZURE_DEPLOYMENT.md` - Complete Azure A100 VM deployment guide
- `AZURE_READY.md` - Azure deployment readiness summary
- `PROJECT_STRUCTURE.md` - Detailed project structure documentation
- `CRITICAL_TEST_RESULTS.md` - Critical test results and validation
- `ALL_TESTS_FIXED.md` - Test fixes and validation summary
- `SFT_DATASET_README.md` - SFT dataset documentation

### 🔧 **Scripts (`scripts/`)**
- `azure_setup.sh` - Complete Azure VM setup
- `azure_quick_start.sh` - One-command Azure deployment
- `run_end_to_end_test.py` - Critical pipeline testing
- `test_pipeline.py` - Alternative testing approach
- `download_test_video.py` - Test video download script

### 🧪 **Tests (`tests/`)**
- `test_ffmpeg_command.sh` - FFmpeg command testing
- `test_inference.py` - Inference testing
- `test_results/` - Test execution results
- `batch_test_results/` - Batch testing results

## 🚀 **Quick Start**

### **Azure A100 VM Deployment**

```bash
# One-command deployment
./scripts/azure_quick_start.sh

# Complete deployment pipeline
make deploy

# Step-by-step deployment
make setup          # Setup environment
make install        # Install dependencies
make test          # Run pipeline tests
make train         # Execute fine-tuning
make evaluate      # Run evaluation
make inference     # Test inference
make validate      # Complete validation
```

### **Local Development**

```bash
# Setup environment
make setup

# Install dependencies
make install

# Run tests
make test

# Start training
make train
```

## 📋 **Phase-by-Phase Workflow**

### **Phase 1: Data Collection**
- Download football videos from various sources
- Organize videos by event classes (Goal, Penalty Shot, Red Card, etc.)
- Target: 10-20 high-quality examples per class

### **Phase 2: Preprocessing**
- Convert videos to 4fps (required by Cosmos-Reason1-7B)
- Resize to standard resolution (720x480)
- Ensure video quality and consistency

### **Phase 3: Annotation**
- Manual annotation of video events
- JSON format with timestamps and descriptions
- Ground truth data for training

### **Phase 4: Dataset Preparation**
- Convert annotations to SFT format
- Create train/validation/test splits
- Prepare (prompt, completion) pairs for training

### **Phase 5: Training**
- LoRA fine-tuning on Cosmos-Reason1-7B
- Azure A100 VM optimized training
- Save LoRA adapter weights

### **Phase 6: Evaluation**
- Test model on hold-out dataset
- Calculate accuracy metrics (Precision, Recall, F1-score)
- Temporal accuracy (tIoU) and description quality (ROUGE, BLEU)

### **Phase 7: Inference**
- Production inference on new videos
- Generate JSON output with event descriptions
- Real-time football video analysis

## 🎯 **Key Features**

### **✅ Complete Pipeline Validation**
- Real video download and 4fps processing
- Manual annotation with correct JSON format
- SFT dataset preparation and validation
- LoRA fine-tuning on Azure A100
- Comprehensive evaluation metrics
- Production-ready inference

### **✅ Azure A100 Optimization**
- DeepSpeed ZeRO-2 optimization
- Mixed precision training (BF16)
- Gradient checkpointing for memory efficiency
- Ray distributed training support
- Weights & Biases monitoring

### **✅ Production Ready**
- Comprehensive error handling
- Detailed logging and monitoring
- Automated deployment scripts
- Complete documentation
- Test coverage and validation

## 📊 **Performance Expectations**

### **Azure A100 Performance**
- **Training Time**: 2-4 hours for full dataset
- **Memory Usage**: 60-70GB GPU memory
- **Batch Size**: 4-8 (depending on video length)
- **Learning Rate**: 2e-5 (optimized for A100)
- **Epochs**: 3 (sufficient for convergence)

### **Accuracy Metrics**
- **Event Classification**: Precision, Recall, F1-score
- **Temporal Accuracy**: Temporal Intersection over Union (tIoU)
- **Description Quality**: ROUGE and BLEU scores

## 🔧 **System Requirements**

### **Azure A100 VM**
- **Instance Type**: Standard_NC24ads_A100_v4
- **Memory**: 220 GB RAM
- **GPU**: 1x A100 (80GB VRAM)
- **Storage**: 1TB SSD
- **OS**: Ubuntu 20.04 LTS

### **Dependencies**
- Python 3.11+
- PyTorch with CUDA support
- Transformers, vLLM, qwen_vl_utils
- DeepSpeed, Ray, Weights & Biases
- FFmpeg, yt-dlp for video processing

## 📚 **Documentation**

### **Complete Documentation Available**
- **Deployment Guide**: `docs/AZURE_DEPLOYMENT.md`
- **Project Structure**: `docs/PROJECT_STRUCTURE.md`
- **Test Results**: `docs/CRITICAL_TEST_RESULTS.md`
- **Azure Ready**: `docs/AZURE_READY.md`

### **Quick Reference**
```bash
# View all available commands
make help

# Check system status
make status

# Monitor performance
make monitor

# Run complete validation
make validate
```

## 🎉 **Success Criteria**

### **Training Success**
- ✅ Loss reduction and convergence
- ✅ LoRA adapter weights saved
- ✅ Evaluation metrics calculated
- ✅ Inference working correctly

### **System Success**
- ✅ GPU utilization >80%
- ✅ Memory usage <70GB
- ✅ Training time <4 hours
- ✅ JSON output generated correctly

## 📞 **Support and Resources**

### **Azure Resources**
- **Azure Documentation**: https://docs.microsoft.com/azure/
- **A100 VM Guide**: https://docs.microsoft.com/azure/virtual-machines/nc-a100-v4-series

### **Project Resources**
- **GitHub Repository**: https://github.com/gns-x/cosmos-football-event-detection
- **Documentation**: Complete deployment guide included
- **Logs**: Detailed logging for troubleshooting
- **Monitoring**: Real-time system monitoring

---

## 🚀 **Ready for Production!**

The Cosmos Football Video Analysis system is ready for production deployment on Azure A100 VMs with complete validation, testing, and documentation.

**Quick Start**: `./scripts/azure_quick_start.sh`
**Complete Deployment**: `make deploy`
**Full Validation**: `make validate`

