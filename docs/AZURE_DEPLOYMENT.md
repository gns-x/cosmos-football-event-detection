# Azure A100 VM Deployment Guide
## Cosmos Football Video Analysis

### 🚀 Complete Azure A100 VM Setup and Training Guide

This guide provides step-by-step instructions for deploying the Cosmos Football Video Analysis system on Azure A100 VMs.

## 📋 Prerequisites

### Azure VM Requirements
- **Instance Type**: Standard_NC24ads_A100_v4
- **Memory**: 220 GB RAM
- **GPU**: 1x A100 (80GB VRAM)
- **Storage**: 1TB SSD
- **OS**: Ubuntu 20.04 LTS
- **Region**: Any region with A100 availability

### Azure Account Setup
1. **Azure Subscription**: Active Azure subscription
2. **Quota**: A100 VM quota approved
3. **Storage**: Azure Storage account (optional)
4. **Networking**: VNet and security groups configured

## 🚀 Quick Start

### 1. Launch Azure A100 VM

```bash
# Create Azure VM using Azure CLI
az vm create \
  --resource-group cosmos-football-rg \
  --name cosmos-football-a100 \
  --image Ubuntu2004 \
  --size Standard_NC24ads_A100_v4 \
  --admin-username azureuser \
  --generate-ssh-keys \
  --storage-sku Premium_LRS \
  --data-disk-sizes-gb 1000
```

### 2. Connect to VM

```bash
# SSH to the VM
ssh azureuser@<VM_IP_ADDRESS>
```

### 3. Clone Repository

```bash
# Clone the project
git clone https://github.com/gns-x/cosmos-football-event-detection.git
cd cosmos-football-event-detection
```

### 4. Run Complete Setup

```bash
# Make setup script executable
chmod +x azure_setup.sh

# Run complete Azure setup
./azure_setup.sh
```

### 5. Activate Environment

```bash
# Activate Python environment
source cosmos-env/bin/activate
```

## 🎯 Complete Deployment Pipeline

### Option 1: Full Automated Deployment

```bash
# Complete deployment with one command
make deploy
```

This will:
- ✅ Setup Azure A100 VM environment
- ✅ Install all dependencies
- ✅ Run critical pipeline tests
- ✅ Execute fine-tuning
- ✅ Run evaluation
- ✅ Test inference
- ✅ Validate complete system

### Option 2: Step-by-Step Deployment

```bash
# 1. Setup environment
make setup

# 2. Install dependencies
make install

# 3. Run tests
make test

# 4. Execute training
make train

# 5. Run evaluation
make evaluate

# 6. Test inference
make inference

# 7. Complete validation
make validate
```

## 📊 System Monitoring

### Check System Status

```bash
# Check overall system status
make status

# Monitor performance
make monitor

# Check GPU status
nvidia-smi
```

### Azure-Specific Monitoring

```bash
# Run Azure monitoring script
./monitor_azure.sh

# Check training progress
tail -f logs/azure_training_*.log
```

## 🎯 Training Configuration

### Azure-Optimized Training

The system includes Azure-specific optimizations:

```toml
# Azure A100 VM Configuration
[azure]
vm_type = "Standard_NC24ads_A100_v4"
gpu_count = 1
gpu_memory = 80
system_memory = 220

# Performance optimizations
[performance]
use_deepspeed = true
use_ray = true
use_wandb = true
mixed_precision = true
gradient_checkpointing = true
```

### Training Execution

```bash
# Start Azure-optimized training
cd 05_training
python azure_training.py --config azure_training_config.toml

# Or use the startup script
./start_azure_training.sh
```

## 📈 Performance Expectations

### Azure A100 Performance

| Metric | Expected Value |
|--------|----------------|
| **Training Time** | 2-4 hours for full dataset |
| **Memory Usage** | 60-70GB GPU memory |
| **Batch Size** | 4-8 (depending on video length) |
| **Learning Rate** | 2e-5 (optimized for A100) |
| **Epochs** | 3 (sufficient for convergence) |

### Monitoring Training

```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Monitor training logs
tail -f logs/azure_training_*.log

# Check Weights & Biases (if enabled)
# Visit wandb.ai to view training metrics
```

## 🔧 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```bash
# Reduce batch size in config
per_device_train_batch_size = 2
gradient_accumulation_steps = 8
```

#### 2. DeepSpeed Issues
```bash
# Check DeepSpeed installation
pip install deepspeed

# Verify DeepSpeed configuration
python -c "import deepspeed; print(deepspeed.__version__)"
```

#### 3. Ray Distributed Training
```bash
# Install Ray
pip install ray[default]

# Check Ray status
ray status
```

### Performance Optimization

#### 1. Memory Optimization
```bash
# Enable gradient checkpointing
use_gradient_checkpointing = true

# Use mixed precision
bf16 = true
fp16 = false
```

#### 2. Training Speed
```bash
# Increase batch size if memory allows
per_device_train_batch_size = 8

# Use gradient accumulation
gradient_accumulation_steps = 2
```

## 📊 Validation and Testing

### Complete Validation Pipeline

```bash
# Run complete validation
make validate
```

This includes:
- ✅ Data pipeline validation
- ✅ Training verification
- ✅ Evaluation metrics
- ✅ Inference testing
- ✅ End-to-end validation

### Manual Testing

```bash
# Test individual components
make test          # Pipeline tests
make train         # Training test
make evaluate      # Evaluation test
make inference     # Inference test
```

## 🎉 Success Criteria

### Training Success
- ✅ **Loss Reduction**: Training loss decreases consistently
- ✅ **Convergence**: Model converges within 3 epochs
- ✅ **Checkpoints**: LoRA adapters saved successfully
- ✅ **Metrics**: Evaluation metrics calculated

### System Success
- ✅ **GPU Utilization**: >80% GPU usage during training
- ✅ **Memory Usage**: <70GB GPU memory usage
- ✅ **Training Time**: <4 hours for full dataset
- ✅ **Inference**: JSON output generated correctly

## 📁 File Structure

```
cosmos-football-event-detection/
├── Makefile                    # Main deployment commands
├── azure_setup.sh             # Azure-specific setup
├── azure_config.yaml          # Azure configuration
├── 05_training/
│   ├── azure_training.py      # Azure-optimized training
│   ├── azure_training_config.toml
│   └── configs/
│       └── azure_deepspeed_config.json
├── logs/                      # Training logs
├── checkpoints/               # Model checkpoints
└── results/                  # Evaluation results
```

## 🚀 Production Deployment

### Final Validation

```bash
# Complete system validation
make validate

# Check all components
make status

# Monitor performance
make monitor
```

### Production Checklist

- ✅ **Environment**: Azure A100 VM configured
- ✅ **Dependencies**: All packages installed
- ✅ **Training**: LoRA model trained successfully
- ✅ **Evaluation**: Accuracy metrics calculated
- ✅ **Inference**: Production-ready JSON output
- ✅ **Monitoring**: System monitoring active

## 📞 Support

### Azure Support
- **Azure Documentation**: https://docs.microsoft.com/azure/
- **A100 VM Guide**: https://docs.microsoft.com/azure/virtual-machines/nc-a100-v4-series
- **Azure Support**: Azure portal support tickets

### Project Support
- **GitHub Issues**: https://github.com/gns-x/cosmos-football-event-detection/issues
- **Documentation**: See project README.md
- **Logs**: Check logs/ directory for detailed information

## 🎯 Next Steps

After successful deployment:

1. **Scale Training**: Use larger datasets for better accuracy
2. **Model Optimization**: Fine-tune hyperparameters
3. **Production Deployment**: Deploy to production environment
4. **Monitoring**: Set up continuous monitoring
5. **Scaling**: Scale to multiple A100 instances if needed

---

**🎉 Azure A100 VM deployment completed successfully!**

The Cosmos Football Video Analysis system is now ready for production use on Azure A100 hardware.
