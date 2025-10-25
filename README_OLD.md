# Cosmos Football Video Analysis Project

This project fine-tunes NVIDIA's Cosmos model for football video analysis and description generation.

## Project Structure

```
/project-cosmos-football/
|
|-- /01_data_collection/
|   |-- download_videos.sh         # Script to download from sources
|   |-- /raw_videos/               # Raw downloaded clips
|
|-- /02_preprocessing/
|   |-- preprocess.sh              # Script using ffmpeg
|   |-- /processed_videos/         # Videos resampled to 4 FPS
|
|-- /03_annotation/
|   |-- /annotation_tool/          # (e.g., CVAT, Vatic, or custom)
|   |-- /ground_truth_json/        # One JSON file per processed video
|
|-- /04_dataset/
|   |-- build_dataset.py           # Script to create train/val/test splits
|   |-- train.jsonl                # Final training file
|   |-- validation.jsonl           # Final validation file
|   |-- test.jsonl                 # Final hold-out test file
|
|-- /05_training/
|   |-- /cosmos-cookbook/          # Git submodule of the NVIDIA repo
|   |-- fine_tune.py               # Your training script (adapting cookbook)
|   |-- config.yaml                # Hyperparameters, paths, LoRA config
|   |-- /checkpoints/              # Saved model adapters (LoRA weights)
|
|-- /06_evaluation/
|   |-- evaluate.py                # Script to run model on test.jsonl
|   |-- /results/                  # Generated JSON outputs
|   |-- metrics.json               # Calculated accuracy (IoU, ROUGE, etc.)
|
|-- /07_inference/
|   |-- inference.py               # Script for running on new videos
|   |-- env_template               # Environment variables template
|   |-- requirements.txt           # Python dependencies
|
|-- README.md                      # Project documentation
```

## Setup Instructions

### 1. Data Collection
```bash
cd 01_data_collection
chmod +x download_videos.sh
./download_videos.sh
```

### 2. Preprocessing
```bash
cd 02_preprocessing
chmod +x preprocess.sh
./preprocess.sh
```

### 3. Annotation
- Set up your preferred annotation tool in `03_annotation/annotation_tool/`
- Create ground truth JSON files in `03_annotation/ground_truth_json/`

### 4. Dataset Creation
```bash
cd 04_dataset
python build_dataset.py
```

### 5. Training
```bash
cd 05_training
# Add cosmos-cookbook as git submodule
git submodule add https://github.com/NVIDIA/cosmos-cookbook.git cosmos-cookbook
python fine_tune.py
```

### 6. Evaluation
```bash
cd 06_evaluation
python evaluate.py
```

### 7. Inference
```bash
cd 07_inference
pip install -r requirements.txt
cp env_template .env
# Edit .env with your configuration
python inference.py --input path/to/video.mp4 --output results.json
```

## Configuration

### Training Configuration (`05_training/config.yaml`)
- Model settings (LoRA rank, learning rate, etc.)
- Data paths and batch sizes
- Hardware configuration

### Inference Configuration (`07_inference/env_template`)
- Model path and device settings
- Inference parameters (temperature, top_p)
- API keys for external services

## Dependencies

### System Requirements
- Python 3.8+
- CUDA-compatible GPU (recommended)
- FFmpeg for video processing

### Python Packages
See `07_inference/requirements.txt` for the complete list of dependencies.

## Usage Examples

### Single Video Inference
```bash
python inference.py --input football_clip.mp4 --output result.json
```

### Batch Processing
```bash
python inference.py --input /path/to/videos/ --batch --output batch_results.json
```

## Model Performance

The fine-tuned model will be evaluated using:
- ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)
- Confidence scores
- Processing time metrics

Results are saved in `06_evaluation/results/metrics.json`.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project follows the same license as the NVIDIA Cosmos cookbook.
