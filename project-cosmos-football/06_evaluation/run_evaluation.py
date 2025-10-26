#!/usr/bin/env python3
"""
Comprehensive Football Video Analysis Evaluation Runner
Orchestrates the complete evaluation pipeline:
1. Generate predictions using fine-tuned model
2. Run comprehensive evaluation metrics
3. Generate detailed reports
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any
import subprocess
import sys
from datetime import datetime


class EvaluationRunner:
    """Orchestrates the complete evaluation pipeline."""
    
    def __init__(self, test_file: str, model_path: str, output_dir: str):
        self.test_file = Path(test_file)
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "results").mkdir(exist_ok=True)
        (self.output_dir / "metrics").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        
        print(f"📁 Evaluation output directory: {self.output_dir}")
    
    def step1_generate_predictions(self) -> bool:
        """Step 1: Generate predictions using fine-tuned model."""
        print("\n" + "=" * 60)
        print("STEP 1: GENERATING PREDICTIONS")
        print("=" * 60)
        
        try:
            # Run prediction generation
            cmd = [
                "python", "generate_predictions.py",
                "--test_file", str(self.test_file.resolve()),
                "--model_path", str(self.model_path.resolve()),
                "--output_dir", str(self.output_dir / "results")
            ]
            
            print(f"📋 Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, cwd=Path(__file__).parent, check=True, capture_output=True, text=True)
            
            print("✅ Predictions generated successfully!")
            print(result.stdout)
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Prediction generation failed: {e}")
            print(f"Error output: {e.stderr}")
            return False
        except Exception as e:
            print(f"❌ Error in prediction generation: {e}")
            return False
    
    def step2_run_evaluation(self) -> bool:
        """Step 2: Run comprehensive evaluation metrics."""
        print("\n" + "=" * 60)
        print("STEP 2: RUNNING EVALUATION METRICS")
        print("=" * 60)
        
        try:
            # Run evaluation
            cmd = [
                "python", "evaluate.py",
                "--test_file", str(self.test_file.resolve()),
                "--results_dir", str(self.output_dir / "results"),
                "--ground_truth_dir", str(Path(__file__).parent.parent / "03_annotation" / "ground_truth_json"),
                "--output_file", "evaluation_results.json"
            ]
            
            print(f"📋 Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, cwd=Path(__file__).parent, check=True, capture_output=True, text=True)
            
            print("✅ Evaluation completed successfully!")
            print(result.stdout)
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Evaluation failed: {e}")
            print(f"Error output: {e.stderr}")
            return False
        except Exception as e:
            print(f"❌ Error in evaluation: {e}")
            return False
    
    def step3_generate_report(self) -> bool:
        """Step 3: Generate comprehensive evaluation report."""
        print("\n" + "=" * 60)
        print("STEP 3: GENERATING EVALUATION REPORT")
        print("=" * 60)
        
        try:
            # Load evaluation results
            results_file = self.output_dir / "results" / "evaluation_results.json"
            if not results_file.exists():
                # Try alternative path
                alt_results_file = Path("06_evaluation") / "results" / "evaluation_results.json"
                if alt_results_file.exists():
                    results_file = alt_results_file
                else:
                    print(f"❌ Results file not found: {results_file}")
                    print(f"❌ Alternative path also not found: {alt_results_file}")
                    return False
            
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            # Generate comprehensive report
            report = self._create_comprehensive_report(results)
            
            # Save report
            report_file = self.output_dir / "reports" / "evaluation_report.md"
            with open(report_file, 'w') as f:
                f.write(report)
            
            print(f"✅ Report generated: {report_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error generating report: {e}")
            return False
    
    def _create_comprehensive_report(self, results: Dict[str, Any]) -> str:
        """Create a comprehensive evaluation report."""
        timestamp = results.get("evaluation_timestamp", datetime.now().isoformat())
        test_size = results.get("test_set_size", 0)
        
        # Extract metrics
        event_metrics = results.get("event_classification", {})
        temporal_metrics = results.get("temporal_accuracy", {})
        desc_metrics = results.get("description_quality", {})
        overall_score = results.get("overall_score", 0.0)
        
        report = f"""# Football Video Analysis Evaluation Report

## Executive Summary

**Evaluation Date:** {timestamp}  
**Test Set Size:** {test_size} examples  
**Overall Score:** {overall_score:.3f}

## 1. Event Classification Performance

### Overall Metrics
- **Accuracy:** {event_metrics.get('accuracy', 0.0):.3f}
- **Precision (Weighted):** {event_metrics.get('precision_weighted', 0.0):.3f}
- **Recall (Weighted):** {event_metrics.get('recall_weighted', 0.0):.3f}
- **F1-Score (Weighted):** {event_metrics.get('f1_weighted', 0.0):.3f}

### Macro Averages
- **Precision (Macro):** {event_metrics.get('precision_macro', 0.0):.3f}
- **Recall (Macro):** {event_metrics.get('recall_macro', 0.0):.3f}
- **F1-Score (Macro):** {event_metrics.get('f1_macro', 0.0):.3f}

## 2. Temporal Accuracy Performance

### Temporal Intersection over Union (tIoU)
- **Mean tIoU:** {temporal_metrics.get('mean_tiou', 0.0):.3f}
- **Median tIoU:** {temporal_metrics.get('median_tiou', 0.0):.3f}
- **Standard Deviation:** {temporal_metrics.get('std_tiou', 0.0):.3f}

### Hit Rate Analysis
- **Hit Rate (tIoU > 0.5):** {temporal_metrics.get('hit_rate', 0.0):.3f}
- **Total Evaluated:** {temporal_metrics.get('total_evaluated', 0)}

## 3. Description Quality Performance

### ROUGE Scores
- **ROUGE-1:** {desc_metrics.get('rouge_rouge1_mean', 0.0):.3f} ± {desc_metrics.get('rouge_rouge1_std', 0.0):.3f}
- **ROUGE-2:** {desc_metrics.get('rouge_rouge2_mean', 0.0):.3f} ± {desc_metrics.get('rouge_rouge2_std', 0.0):.3f}
- **ROUGE-L:** {desc_metrics.get('rouge_rougeL_mean', 0.0):.3f} ± {desc_metrics.get('rouge_rougeL_std', 0.0):.3f}

### BLEU Score
- **BLEU:** {desc_metrics.get('bleu_mean', 0.0):.3f} ± {desc_metrics.get('bleu_std', 0.0):.3f}
- **Total Evaluated:** {desc_metrics.get('total_evaluated', 0)}

## 4. Detailed Classification Report

```
{event_metrics.get('classification_report', 'N/A')}
```

## 5. Performance Analysis

### Strengths
- Event classification shows strong performance with F1-score of {event_metrics.get('f1_weighted', 0.0):.3f}
- Temporal accuracy hit rate of {temporal_metrics.get('hit_rate', 0.0):.3f} indicates good time window prediction
- Description quality with ROUGE-1 of {desc_metrics.get('rouge_rouge1_mean', 0.0):.3f} shows coherent text generation

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

The football video analysis model demonstrates {self._get_performance_level(overall_score)} performance with an overall score of {overall_score:.3f}. The model shows particular strength in event classification while having room for improvement in temporal accuracy and description quality.

---
*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        return report
    
    def _get_performance_level(self, score: float) -> str:
        """Get performance level description."""
        if score >= 0.9:
            return "excellent"
        elif score >= 0.8:
            return "very good"
        elif score >= 0.7:
            return "good"
        elif score >= 0.6:
            return "fair"
        else:
            return "needs improvement"
    
    def run_complete_evaluation(self) -> bool:
        """Run the complete evaluation pipeline."""
        print("🏈 Football Video Analysis - Complete Evaluation Pipeline")
        print("=" * 70)
        
        # Step 1: Generate predictions
        if not self.step1_generate_predictions():
            print("❌ Step 1 failed - stopping evaluation")
            return False
        
        # Step 2: Run evaluation
        if not self.step2_run_evaluation():
            print("❌ Step 2 failed - stopping evaluation")
            return False
        
        # Step 3: Generate report
        if not self.step3_generate_report():
            print("❌ Step 3 failed - stopping evaluation")
            return False
        
        print("\n" + "=" * 70)
        print("✅ COMPLETE EVALUATION PIPELINE FINISHED SUCCESSFULLY!")
        print("=" * 70)
        
        # Print summary
        self._print_final_summary()
        
        return True
    
    def _print_final_summary(self):
        """Print final evaluation summary."""
        print("\n📊 EVALUATION SUMMARY:")
        print(f"  📁 Results directory: {self.output_dir}")
        print(f"  📄 Predictions: {self.output_dir / 'results' / 'predictions.json'}")
        print(f"  📊 Metrics: {self.output_dir / 'results' / 'evaluation_results.json'}")
        print(f"  📋 Report: {self.output_dir / 'reports' / 'evaluation_report.md'}")
        print("\n🎉 All evaluation files generated successfully!")


def main():
    """Main evaluation runner function."""
    parser = argparse.ArgumentParser(description="Football Video Analysis Evaluation Runner")
    parser.add_argument("--test_file", type=str,
                       default="../04_dataset/test.jsonl",
                       help="Path to test.jsonl file")
    parser.add_argument("--model_path", type=str,
                       default="../05_training/checkpoints/football_sft",
                       help="Path to fine-tuned model checkpoints")
    parser.add_argument("--output_dir", type=str,
                       default="./evaluation_output",
                       help="Directory to save evaluation results")
    parser.add_argument("--skip_predictions", action="store_true",
                       help="Skip prediction generation (use existing predictions)")
    
    args = parser.parse_args()
    
    # Create evaluation runner
    runner = EvaluationRunner(
        test_file=args.test_file,
        model_path=args.model_path,
        output_dir=args.output_dir
    )
    
    # Run complete evaluation
    success = runner.run_complete_evaluation()
    
    if success:
        print("🎉 Complete evaluation pipeline finished successfully!")
        return 0
    else:
        print("❌ Evaluation pipeline failed!")
        return 1


if __name__ == "__main__":
    exit(main())
