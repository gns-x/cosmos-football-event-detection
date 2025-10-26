#!/usr/bin/env python3
"""
Comprehensive Football Video Analysis Evaluation Script
Measures accuracy across multiple dimensions:
1. Event Classification (Precision, Recall, F1-score)
2. Temporal Accuracy (Temporal IoU)
3. Description Quality (ROUGE, BLEU)
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
import re

# Evaluation metrics
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass


class FootballVideoEvaluator:
    """Comprehensive evaluator for football video analysis."""
    
    def __init__(self, test_file: str, results_dir: str, ground_truth_dir: str):
        self.test_file = Path(test_file)
        self.results_dir = Path(results_dir)
        self.ground_truth_dir = Path(ground_truth_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.bleu_smoothing = SmoothingFunction().method1
        
        # Load test data
        self.test_data = self._load_test_data()
        self.ground_truth_data = self._load_ground_truth()
        
        print(f"📊 Loaded {len(self.test_data)} test examples")
        print(f"📊 Loaded {len(self.ground_truth_data)} ground truth annotations")
    
    def _load_test_data(self) -> List[Dict[str, Any]]:
        """Load test dataset."""
        test_data = []
        with open(self.test_file, 'r') as f:
            for line in f:
                if line.strip():
                    test_data.append(json.loads(line))
        return test_data
    
    def _load_ground_truth(self) -> Dict[str, Any]:
        """Load ground truth annotations."""
        ground_truth = {}
        for gt_file in self.ground_truth_dir.glob("*.json"):
            with open(gt_file, 'r') as f:
                data = json.load(f)
                # Handle both array and object formats
                if isinstance(data, list):
                    # Array format - use filename as video name
                    video_name = gt_file.stem
                    ground_truth[video_name] = {
                        "video_file": f"{video_name}.mp4",
                        "annotations": data
                    }
                else:
                    # Object format - use video_file field
                    video_name = data.get("video_file", gt_file.stem)
                    ground_truth[video_name] = data
        return ground_truth
    
    def parse_time_to_seconds(self, time_str: str) -> float:
        """Parse time string (MM:SS) to seconds."""
        if not time_str:
            return 0.0
        
        # Handle MM:SS format
        if ':' in time_str:
            parts = time_str.split(':')
            if len(parts) == 2:
                minutes, seconds = parts
                return int(minutes) * 60 + int(seconds)
            elif len(parts) == 3:
                hours, minutes, seconds = parts
                return int(hours) * 3600 + int(minutes) * 60 + int(seconds)
        
        # Handle seconds only
        try:
            return float(time_str)
        except:
            return 0.0
    
    def calculate_temporal_iou(self, pred_start: str, pred_end: str, gt_start: str, gt_end: str) -> float:
        """Calculate Temporal Intersection over Union (tIoU)."""
        pred_start_sec = self.parse_time_to_seconds(pred_start)
        pred_end_sec = self.parse_time_to_seconds(pred_end)
        gt_start_sec = self.parse_time_to_seconds(gt_start)
        gt_end_sec = self.parse_time_to_seconds(gt_end)
        
        # Calculate intersection
        intersection_start = max(pred_start_sec, gt_start_sec)
        intersection_end = min(pred_end_sec, gt_end_sec)
        intersection = max(0, intersection_end - intersection_start)
        
        # Calculate union
        pred_duration = pred_end_sec - pred_start_sec
        gt_duration = gt_end_sec - gt_start_sec
        union = pred_duration + gt_duration - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def evaluate_event_classification(self, predictions: List[Dict], ground_truth: List[Dict]) -> Dict[str, float]:
        """Evaluate event classification accuracy."""
        print("🎯 Evaluating Event Classification...")
        
        # Extract predicted and ground truth events
        pred_events = []
        gt_events = []
        
        for pred, gt in zip(predictions, ground_truth):
            pred_events.append(pred.get("event", "Unknown"))
            gt_events.append(gt.get("event", "Unknown"))
        
        # Calculate metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            gt_events, pred_events, average='weighted', zero_division=0
        )
        
        # Per-class metrics
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            gt_events, pred_events, average='macro', zero_division=0
        )
        
        # Overall accuracy
        accuracy = sum(1 for p, g in zip(pred_events, gt_events) if p == g) / len(pred_events)
        
        return {
            "accuracy": accuracy,
            "precision_weighted": precision,
            "recall_weighted": recall,
            "f1_weighted": f1,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "classification_report": classification_report(gt_events, pred_events, zero_division=0)
        }
    
    def evaluate_temporal_accuracy(self, predictions: List[Dict], ground_truth: List[Dict]) -> Dict[str, float]:
        """Evaluate temporal accuracy using tIoU."""
        print("⏰ Evaluating Temporal Accuracy...")
        
        tious = []
        hits = 0  # tIoU > 0.5
        
        for pred, gt in zip(predictions, ground_truth):
            pred_start = pred.get("start_time", "")
            pred_end = pred.get("end_time", "")
            gt_start = gt.get("start_time", "")
            gt_end = gt.get("end_time", "")
            
            if pred_start and pred_end and gt_start and gt_end:
                tiou = self.calculate_temporal_iou(pred_start, pred_end, gt_start, gt_end)
                tious.append(tiou)
                
                if tiou > 0.5:
                    hits += 1
        
        if not tious:
            return {"mean_tiou": 0.0, "hit_rate": 0.0, "total_evaluated": 0}
        
        return {
            "mean_tiou": np.mean(tious),
            "median_tiou": np.median(tious),
            "std_tiou": np.std(tious),
            "hit_rate": hits / len(tious),
            "total_evaluated": len(tious)
        }
    
    def evaluate_description_quality(self, predictions: List[Dict], ground_truth: List[Dict]) -> Dict[str, float]:
        """Evaluate description quality using ROUGE and BLEU."""
        print("📝 Evaluating Description Quality...")
        
        rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
        bleu_scores = []
        
        for pred, gt in zip(predictions, ground_truth):
            pred_desc = pred.get("description", "")
            gt_desc = gt.get("description", "")
            
            if pred_desc and gt_desc:
                # ROUGE scores
                rouge_scores_dict = self.rouge_scorer.score(gt_desc, pred_desc)
                for metric in rouge_scores:
                    rouge_scores[metric].append(rouge_scores_dict[metric].fmeasure)
                
                # BLEU score
                try:
                    # Tokenize for BLEU
                    pred_tokens = nltk.word_tokenize(pred_desc.lower())
                    gt_tokens = nltk.word_tokenize(gt_desc.lower())
                    
                    bleu = sentence_bleu([gt_tokens], pred_tokens, smoothing_function=self.bleu_smoothing)
                    bleu_scores.append(bleu)
                except:
                    bleu_scores.append(0.0)
        
        # Calculate averages
        results = {}
        for metric in rouge_scores:
            if rouge_scores[metric]:
                results[f"rouge_{metric}_mean"] = np.mean(rouge_scores[metric])
                results[f"rouge_{metric}_std"] = np.std(rouge_scores[metric])
            else:
                results[f"rouge_{metric}_mean"] = 0.0
                results[f"rouge_{metric}_std"] = 0.0
        
        if bleu_scores:
            results["bleu_mean"] = np.mean(bleu_scores)
            results["bleu_std"] = np.std(bleu_scores)
        else:
            results["bleu_mean"] = 0.0
            results["bleu_std"] = 0.0
        
        results["total_evaluated"] = len(bleu_scores)
        
        return results
    
    def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """Run comprehensive evaluation across all dimensions."""
        print("🚀 Starting Comprehensive Football Video Analysis Evaluation")
        print("=" * 70)
        
        # For now, we'll create mock predictions since we don't have the trained model yet
        # In a real scenario, you would load predictions from the fine-tuned model
        predictions = self._create_mock_predictions()
        
        # Run evaluations
        event_metrics = self.evaluate_event_classification(predictions, self.test_data)
        temporal_metrics = self.evaluate_temporal_accuracy(predictions, self.test_data)
        description_metrics = self.evaluate_description_quality(predictions, self.test_data)
        
        # Compile results
        results = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "test_set_size": len(self.test_data),
            "event_classification": event_metrics,
            "temporal_accuracy": temporal_metrics,
            "description_quality": description_metrics,
            "overall_score": self._calculate_overall_score(event_metrics, temporal_metrics, description_metrics)
        }
        
        return results
    
    def _create_mock_predictions(self) -> List[Dict[str, Any]]:
        """Create mock predictions for testing (replace with actual model predictions)."""
        predictions = []
        
        for test_item in self.test_data:
            # Mock prediction - in reality, this would come from your fine-tuned model
            video_name = Path(test_item["video_path"]).name
            class_name = test_item["class"].replace("_", " ").title()
            
            prediction = {
                "video": test_item["video_path"],
                "event": class_name,
                "description": f"Mock prediction for {class_name} event in {video_name}",
                "start_time": "0:05",
                "end_time": "0:15",
                "confidence": 0.85
            }
            predictions.append(prediction)
        
        return predictions
    
    def _calculate_overall_score(self, event_metrics: Dict, temporal_metrics: Dict, description_metrics: Dict) -> float:
        """Calculate overall evaluation score."""
        # Weighted combination of all metrics
        event_score = event_metrics.get("f1_weighted", 0.0)
        temporal_score = temporal_metrics.get("hit_rate", 0.0)
        description_score = description_metrics.get("rouge_rouge1_mean", 0.0)
        
        # Weighted average (adjust weights as needed)
        overall_score = (0.4 * event_score + 0.3 * temporal_score + 0.3 * description_score)
        
        return overall_score
    
    def save_results(self, results: Dict[str, Any], output_file: str):
        """Save evaluation results to file."""
        output_path = self.results_dir / output_file
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Results saved to: {output_path}")
    
    def generate_report(self, results: Dict[str, Any]):
        """Generate a comprehensive evaluation report."""
        print("\n" + "=" * 70)
        print("📊 FOOTBALL VIDEO ANALYSIS EVALUATION REPORT")
        print("=" * 70)
        
        # Overall Score
        overall_score = results.get("overall_score", 0.0)
        print(f"\n🎯 OVERALL SCORE: {overall_score:.3f}")
        
        # Event Classification
        event_metrics = results.get("event_classification", {})
        print(f"\n📈 EVENT CLASSIFICATION:")
        print(f"  Accuracy: {event_metrics.get('accuracy', 0.0):.3f}")
        print(f"  Precision (Weighted): {event_metrics.get('precision_weighted', 0.0):.3f}")
        print(f"  Recall (Weighted): {event_metrics.get('recall_weighted', 0.0):.3f}")
        print(f"  F1-Score (Weighted): {event_metrics.get('f1_weighted', 0.0):.3f}")
        
        # Temporal Accuracy
        temporal_metrics = results.get("temporal_accuracy", {})
        print(f"\n⏰ TEMPORAL ACCURACY:")
        print(f"  Mean tIoU: {temporal_metrics.get('mean_tiou', 0.0):.3f}")
        print(f"  Hit Rate (tIoU > 0.5): {temporal_metrics.get('hit_rate', 0.0):.3f}")
        print(f"  Total Evaluated: {temporal_metrics.get('total_evaluated', 0)}")
        
        # Description Quality
        desc_metrics = results.get("description_quality", {})
        print(f"\n📝 DESCRIPTION QUALITY:")
        print(f"  ROUGE-1: {desc_metrics.get('rouge_rouge1_mean', 0.0):.3f}")
        print(f"  ROUGE-2: {desc_metrics.get('rouge_rouge2_mean', 0.0):.3f}")
        print(f"  ROUGE-L: {desc_metrics.get('rouge_rougeL_mean', 0.0):.3f}")
        print(f"  BLEU: {desc_metrics.get('bleu_mean', 0.0):.3f}")
        
        print("\n" + "=" * 70)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Football Video Analysis Evaluation")
    parser.add_argument("--test_file", type=str, 
                       default="../04_dataset/test.jsonl",
                       help="Path to test.jsonl file")
    parser.add_argument("--results_dir", type=str,
                       default="./results",
                       help="Directory to save evaluation results")
    parser.add_argument("--ground_truth_dir", type=str,
                       default="../03_annotation/ground_truth_json",
                       help="Directory containing ground truth annotations")
    parser.add_argument("--output_file", type=str,
                       default="evaluation_results.json",
                       help="Output file for results")
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = FootballVideoEvaluator(
        test_file=args.test_file,
        results_dir=args.results_dir,
        ground_truth_dir=args.ground_truth_dir
    )
    
    # Run evaluation
    results = evaluator.run_comprehensive_evaluation()
    
    # Save results
    evaluator.save_results(results, args.output_file)
    
    # Generate report
    evaluator.generate_report(results)
    
    print("\n✅ Evaluation completed successfully!")


if __name__ == "__main__":
    main()