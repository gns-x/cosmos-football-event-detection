# Project Structure
## Cosmos Football Video Analysis

This document describes the organized project structure following best practices for machine learning projects.

## 📁 **Root Directory Structure**

```
cosmos-football-event-detection/
├── README.md                           # Main project README
├── Makefile                           # Deployment and automation commands
├── environment.yml                    # Conda environment specification
├── activate_env.sh                    # Environment activation script
├── docs/                              # 📚 All documentation files
├── scripts/                           # 🔧 All executable scripts
├── tests/                             # 🧪 All test files and results
├── 01_data_collection/               # Data collection phase
├── 02_preprocessing/                 # Video preprocessing phase
├── 03_annotation/                    # Manual annotation phase
├── 04_dataset/                       # Dataset preparation phase
├── 05_training/                      # Model training phase
├── 06_evaluation/                    # Model evaluation phase
└── 07_inference/                     # Model inference phase
```

## 📚 **Documentation (`docs/`)**

All markdown documentation files (except README files) are organized in the `docs/` folder:

```
docs/
├── PROJECT_STRUCTURE.md              # This file
├── AZURE_DEPLOYMENT.md               # Azure A100 VM deployment guide
├── AZURE_READY.md                    # Azure deployment readiness summary
├── ALL_TESTS_FIXED.md                # Test fixes documentation
├── CRITICAL_TEST_RESULTS.md          # Critical test results
├── SFT_DATASET_README.md             # SFT dataset documentation
├── evaluation_report.md              # Evaluation results report
└── [other documentation files]
```

## 🔧 **Scripts (`scripts/`)**

All executable scripts are organized in the `scripts/` folder:

```
scripts/
├── azure_setup.sh                    # Complete Azure VM setup
├── azure_quick_start.sh              # One-command Azure deployment
├── run_end_to_end_test.py            # Critical pipeline testing
├── test_pipeline.py                  # Alternative testing
└── download_test_video.py            # Test video download script
```

## 🧪 **Tests (`tests/`)**

All test files and test results are organized in the `tests/` folder:

```
tests/
├── test_ffmpeg_command.sh            # FFmpeg command testing
├── test_inference.py                 # Inference testing
├── test_results/                     # Test execution results
│   └── test_analysis_result.json
└── batch_test_results/               # Batch testing results
    ├── batch_analysis_results.json
    ├── goal_sample_1_processed_analysis.json
    ├── penalty_shot_sample_1_processed_analysis.json
    └── red_card_sample_1_processed_analysis.json
```

## 🎯 **Phase-Based Organization**

The project follows a clear phase-based structure for the machine learning pipeline:

### **Phase 1: Data Collection (`01_data_collection/`)**
```
01_data_collection/
├── download_videos.sh                # Video download script
├── collect_sample_data.py            # Sample data collection
└── raw_videos/                       # Raw video files
    ├── goal/
    ├── penalty_shot/
    ├── red_card/
    └── [other classes]
```

### **Phase 2: Preprocessing (`02_preprocessing/`)**
```
02_preprocessing/
├── preprocess.sh                     # Video preprocessing script
└── processed_videos/                 # Processed video files
    ├── goal/
    ├── penalty_shot/
    ├── red_card/
    └── [other classes]
```

### **Phase 3: Annotation (`03_annotation/`)**
```
03_annotation/
├── setup_annotation_tool.py         # Annotation tool setup
├── create_ground_truth.py            # Ground truth creation
├── manual_annotation.py              # Manual annotation script
├── annotation_tool/                  # Web-based annotation tool
└── ground_truth_json/                # Ground truth JSON files
```

### **Phase 4: Dataset (`04_dataset/`)**
```
04_dataset/
├── build_dataset.py                  # Dataset building script
├── build_sft_dataset.py              # SFT dataset building
├── train.jsonl                       # Training dataset
├── validation.jsonl                  # Validation dataset
└── test.jsonl                        # Test dataset
```

### **Phase 5: Training (`05_training/`)**
```
05_training/
├── azure_training.py                 # Azure-optimized training
├── azure_training_config.toml        # Azure training configuration
├── football_sft.py                   # Football SFT training
├── fine_tune.py                      # Fine-tuning script
├── configs/                          # Training configurations
│   ├── azure_deepspeed_config.json
│   └── deepspeed_config.json
├── checkpoints/                      # Model checkpoints
└── cosmos-cookbook/                  # NVIDIA Cosmos Cookbook submodule
```

### **Phase 6: Evaluation (`06_evaluation/`)**
```
06_evaluation/
├── evaluate.py                       # Evaluation script
├── generate_predictions.py           # Prediction generation
├── run_evaluation.py                 # Evaluation runner
├── results/                          # Evaluation results
├── metrics/                          # Evaluation metrics
└── reports/                          # Evaluation reports
```

### **Phase 7: Inference (`07_inference/`)**
```
07_inference/
├── football_inference.py             # Production inference
├── simple_inference.py              # Simplified inference
├── inference.py                     # Basic inference script
└── requirements.txt                 # Inference dependencies
```

## 🚀 **Deployment Commands**

The project includes comprehensive deployment automation:

```bash
# Quick deployment
make deploy

# Step-by-step deployment
make setup          # Setup environment
make install        # Install dependencies
make test          # Run pipeline tests
make train         # Execute fine-tuning
make evaluate      # Run evaluation
make inference     # Test inference
make validate      # Complete validation

# Azure-specific deployment
./scripts/azure_quick_start.sh
```

## 📊 **File Organization Principles**

### **1. Documentation Separation**
- All `.md` files (except README files) go in `docs/`
- README files stay in their respective directories
- Documentation is centralized and easily accessible

### **2. Script Centralization**
- All executable scripts go in `scripts/`
- Scripts are organized by functionality
- Easy to find and execute

### **3. Test Organization**
- All test files go in `tests/`
- Test results are organized in subdirectories
- Clear separation of test code and results

### **4. Phase-Based Structure**
- Each phase has its own directory
- Clear progression through the pipeline
- Easy to understand the workflow

## 🔧 **Maintenance and Updates**

### **Adding New Documentation**
```bash
# Add new documentation
mv new_documentation.md docs/
```

### **Adding New Scripts**
```bash
# Add new scripts
mv new_script.py scripts/
chmod +x scripts/new_script.py
```

### **Adding New Tests**
```bash
# Add new tests
mv new_test.py tests/
```

### **Updating References**
When moving files, update references in:
- `Makefile`
- Script imports
- Configuration files
- Documentation links

## 📋 **Best Practices**

### **1. File Naming**
- Use descriptive names
- Use snake_case for Python files
- Use kebab-case for documentation
- Use UPPERCASE for configuration files

### **2. Directory Structure**
- Keep related files together
- Use clear, descriptive directory names
- Follow the established phase-based structure

### **3. Documentation**
- Keep documentation up to date
- Use consistent formatting
- Include examples and usage instructions

### **4. Scripts**
- Make scripts executable
- Include proper error handling
- Add usage documentation
- Use consistent coding style

## 🎯 **Benefits of This Structure**

1. **Clear Organization**: Easy to find files and understand the project
2. **Scalability**: Easy to add new components and phases
3. **Maintainability**: Clear separation of concerns
4. **Collaboration**: Team members can easily navigate the project
5. **Deployment**: Automated deployment with clear commands
6. **Testing**: Organized test structure with clear results
7. **Documentation**: Centralized and accessible documentation

---

**This project structure follows industry best practices for machine learning projects and provides a clear, organized, and maintainable codebase.**

