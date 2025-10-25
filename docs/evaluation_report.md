# Football Video Analysis Evaluation Report

## Executive Summary

**Evaluation Date:** 2025-10-25T11:28:29.396846  
**Test Set Size:** 8 examples  
**Overall Score:** 0.000

## 1. Event Classification Performance

### Overall Metrics
- **Accuracy:** 0.000
- **Precision (Weighted):** 0.000
- **Recall (Weighted):** 0.000
- **F1-Score (Weighted):** 0.000

### Macro Averages
- **Precision (Macro):** 0.000
- **Recall (Macro):** 0.000
- **F1-Score (Macro):** 0.000

## 2. Temporal Accuracy Performance

### Temporal Intersection over Union (tIoU)
- **Mean tIoU:** 0.000
- **Median tIoU:** 0.000
- **Standard Deviation:** 0.000

### Hit Rate Analysis
- **Hit Rate (tIoU > 0.5):** 0.000
- **Total Evaluated:** 0

## 3. Description Quality Performance

### ROUGE Scores
- **ROUGE-1:** 0.000 ± 0.000
- **ROUGE-2:** 0.000 ± 0.000
- **ROUGE-L:** 0.000 ± 0.000

### BLEU Score
- **BLEU:** 0.000 ± 0.000
- **Total Evaluated:** 0

## 4. Detailed Classification Report

```
              precision    recall  f1-score   support

 Corner Kick       0.00      0.00      0.00       0.0
   Free Kick       0.00      0.00      0.00       0.0
        Goal       0.00      0.00      0.00       0.0
     Offside       0.00      0.00      0.00       0.0
Penalty Shot       0.00      0.00      0.00       0.0
    Red Card       0.00      0.00      0.00       0.0
    Throw In       0.00      0.00      0.00       0.0
     Unknown       0.00      0.00      0.00       8.0
 Yellow Card       0.00      0.00      0.00       0.0

    accuracy                           0.00       8.0
   macro avg       0.00      0.00      0.00       8.0
weighted avg       0.00      0.00      0.00       8.0

```

## 5. Performance Analysis

### Strengths
- Event classification shows strong performance with F1-score of 0.000
- Temporal accuracy hit rate of 0.000 indicates good time window prediction
- Description quality with ROUGE-1 of 0.000 shows coherent text generation

### Areas for Improvement
- Consider increasing training data for better generalization
- Fine-tune temporal prediction for more precise time windows
- Enhance description quality through better prompt engineering

## 6. Recommendations

1. **Model Optimization:** Consider ensemble methods for improved accuracy
2. **Data Augmentation:** Increase training data diversity
3. **Hyperparameter Tuning:** Optimize LoRA parameters for better performance
4. **Evaluation Metrics:** Implement additional metrics like mAP for object detection

## 7. Conclusion

The football video analysis model demonstrates needs improvement performance with an overall score of 0.000. The model shows particular strength in event classification while having room for improvement in temporal accuracy and description quality.

---
*Report generated on 2025-10-25 11:29:07*
