
# 🏆 Cosmos Football Video Dataset Summary

## 📊 Dataset Statistics
- **Total Samples**: 24
- **Training**: 16 samples (66.7%)
- **Validation**: 0 samples (0.0%)
- **Test**: 8 samples (33.3%)

## 🏷️ Classes (8)
- corner_kick
- free_kick
- goal
- offside
- penalty_shot
- red_card
- throw_in
- yellow_card

## 📁 Files
- `train.jsonl` - Training data
- `validation.jsonl` - Validation data  
- `test.jsonl` - Test data
- `dataset_metadata.json` - Dataset metadata

## 🎯 Cosmos Format
- **Video FPS**: 4 (required by Cosmos-Reason1-7B)
- **Max Duration**: 30 seconds
- **Resolution**: 720x480
- **Format**: JSONL with video paths and annotations

## 🚀 Next Steps
1. Review the dataset splits
2. Start training with `05_training/fine_tune.py`
3. Evaluate with `06_evaluation/evaluate.py`
