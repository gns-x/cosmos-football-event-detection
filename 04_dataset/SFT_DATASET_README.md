
# 🏆 Cosmos Football Video SFT Dataset Summary

## 📊 Dataset Statistics
- **Total Examples**: 17
- **Training**: 13 examples (76.5%)
- **Validation**: 4 examples (23.5%)

## 🏷️ Classes (0)


## 📁 Files
- `train.jsonl` - Training data in SFT format
- `validation.jsonl` - Validation data in SFT format
- `sft_dataset_metadata.json` - Dataset metadata

## 🎯 SFT Format
- **Prompt**: "Analyze the following football clip. Identify all significant events including goals, cards, and shots. For each event, provide a detailed description including player, team, and jersey info, the event class, and precise start/end timestamps. Output *only* a valid JSON array."
- **Completion**: JSON array of event annotations
- **Video FPS**: 4 (required by Cosmos-Reason1-7B)
- **Max Duration**: 30 seconds
- **Resolution**: 720x480

## 📝 Example SFT Entry
```json
{
  "video": "02_preprocessing/processed_videos/goal/goal_sample_1_processed.mp4",
  "prompt": "Analyze the following football clip. Identify all significant events including goals, cards, and shots. For each event, provide a detailed description including player, team, and jersey info, the event class, and precise start/end timestamps. Output *only* a valid JSON array.",
  "completion": "[{\"description\": \"Player scores a goal in the goal video. The ball crosses the goal line and the referee signals a goal.\", \"start_time\": \"0:00:05\", \"end_time\": \"0:00:15\", \"event\": \"Goal\"}]"
}
```

## 🚀 Next Steps
1. Review the SFT dataset format
2. Start training with `05_training/fine_tune.py`
3. Evaluate with `06_evaluation/evaluate.py`
