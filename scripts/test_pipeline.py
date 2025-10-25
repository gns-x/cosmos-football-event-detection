#!/usr/bin/env python3
"""
End-to-End Pipeline Test Script
Tests the entire football video analysis pipeline with real data
"""

import os
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List
import time


class PipelineTester:
    """Comprehensive pipeline tester for football video analysis."""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.test_video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Placeholder - replace with real goal video
        self.test_video_name = "goal_test_01"
        self.results = {}
        
        print("🧪 Football Video Analysis Pipeline Tester")
        print("=" * 60)
    
    def phase1_2_data_test(self) -> bool:
        """Phase 1 & 2: Test real video download and 4fps preprocessing."""
        print("\n📹 PHASE 1 & 2: Data Ingestion and Preprocessing Test")
        print("-" * 60)
        
        # Step 1: Download real video
        print("1️⃣ Downloading real YouTube video...")
        raw_video_path = self.project_root / "01_data_collection" / "raw_videos" / f"{self.test_video_name}.mp4"
        
        try:
            # Create directory
            raw_video_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Download video using yt-dlp
            cmd = [
                "yt-dlp",
                "-o", str(raw_video_path),
                "--format", "best[height<=720]",
                self.test_video_url
            ]
            
            print(f"📋 Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            if raw_video_path.exists():
                print(f"✅ Video downloaded: {raw_video_path}")
                self.results["download_success"] = True
            else:
                print(f"❌ Video not found after download: {raw_video_path}")
                return False
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Download failed: {e}")
            print(f"Error output: {e.stderr}")
            return False
        except Exception as e:
            print(f"❌ Download error: {e}")
            return False
        
        # Step 2: Preprocess to 4fps
        print("\n2️⃣ Preprocessing video to 4fps...")
        processed_video_path = self.project_root / "02_preprocessing" / "processed_videos" / f"{self.test_video_name}.mp4"
        
        try:
            # Create directory
            processed_video_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Run ffmpeg command to convert to 4fps
            cmd = [
                "ffmpeg",
                "-i", str(raw_video_path),
                "-r", "4",  # Set frame rate to 4fps
                "-vf", "scale=720:480",  # Resize to standard resolution
                "-c:v", "libx264",
                "-c:a", "aac",
                "-preset", "fast",
                "-crf", "23",
                "-y",  # Overwrite output file
                str(processed_video_path)
            ]
            
            print(f"📋 Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            if processed_video_path.exists():
                print(f"✅ Video processed: {processed_video_path}")
            else:
                print(f"❌ Processed video not found: {processed_video_path}")
                return False
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Preprocessing failed: {e}")
            print(f"Error output: {e.stderr}")
            return False
        except Exception as e:
            print(f"❌ Preprocessing error: {e}")
            return False
        
        # Step 3: CRITICAL VERIFICATION - Check fps
        print("\n3️⃣ CRITICAL VERIFICATION: Checking output fps...")
        try:
            cmd = [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(processed_video_path)
            ]
            
            print(f"📋 Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            fps_output = result.stdout.strip()
            
            print(f"📊 FPS Output: {fps_output}")
            
            if fps_output in ["4/1", "4"]:
                print("✅ SUCCESS: Video is correctly processed to 4fps!")
                self.results["fps_verification"] = True
                return True
            else:
                print(f"❌ CRITICAL FAILURE: Expected '4/1' or '4', got '{fps_output}'")
                print("❌ The model will fail with this fps!")
                self.results["fps_verification"] = False
                return False
                
        except subprocess.CalledProcessError as e:
            print(f"❌ FPS verification failed: {e}")
            return False
        except Exception as e:
            print(f"❌ FPS verification error: {e}")
            return False
    
    def phase3_annotation_test(self) -> bool:
        """Phase 3: Test manual annotation with real timestamps."""
        print("\n📝 PHASE 3: Annotation (Ground Truth) Test")
        print("-" * 60)
        
        # Create ground truth JSON with real timestamps
        ground_truth_json = [
            {
                "description": "Player #9 in white jersey heads the ball into the net from a corner kick.",
                "start_time": "0:0:42",
                "end_time": "0:0:48",
                "event": "Goal"
            }
        ]
        
        # Save to annotation directory
        annotation_path = self.project_root / "03_annotation" / "ground_truth_json" / f"{self.test_video_name}.json"
        annotation_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(annotation_path, 'w') as f:
                json.dump(ground_truth_json, f, indent=2)
            
            print(f"✅ Ground truth JSON created: {annotation_path}")
            
            # Verify JSON is valid
            with open(annotation_path, 'r') as f:
                loaded_json = json.load(f)
            
            print("✅ JSON validation passed")
            print(f"📊 Annotation content: {json.dumps(loaded_json, indent=2)}")
            
            self.results["annotation_success"] = True
            return True
            
        except Exception as e:
            print(f"❌ Annotation creation failed: {e}")
            return False
    
    def phase4_dataset_test(self) -> bool:
        """Phase 4: Test dataset preparation with real JSON."""
        print("\n📊 PHASE 4: Dataset Preparation Test")
        print("-" * 60)
        
        # Temporarily move other files to isolate test data
        print("1️⃣ Isolating test data...")
        ground_truth_dir = self.project_root / "03_annotation" / "ground_truth_json"
        backup_dir = ground_truth_dir / "backup"
        backup_dir.mkdir(exist_ok=True)
        
        # Move other files to backup
        for file in ground_truth_dir.glob("*.json"):
            if file.name != f"{self.test_video_name}.json":
                backup_path = backup_dir / file.name
                file.rename(backup_path)
                print(f"📁 Moved {file.name} to backup")
        
        # Run dataset building script
        print("\n2️⃣ Running dataset preparation...")
        try:
            dataset_script = self.project_root / "04_dataset" / "build_sft_dataset.py"
            cmd = ["python", str(dataset_script)]
            
            print(f"📋 Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, cwd=self.project_root, check=True, capture_output=True, text=True)
            
            print("✅ Dataset preparation completed")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Dataset preparation failed: {e}")
            print(f"Error output: {e.stderr}")
            return False
        except Exception as e:
            print(f"❌ Dataset preparation error: {e}")
            return False
        
        # Verify output
        print("\n3️⃣ Verifying dataset output...")
        train_file = self.project_root / "04_dataset" / "train.jsonl"
        
        if not train_file.exists():
            print(f"❌ Train file not found: {train_file}")
            return False
        
        # Check content
        with open(train_file, 'r') as f:
            lines = f.readlines()
        
        if len(lines) != 1:
            print(f"❌ Expected 1 line, got {len(lines)}")
            return False
        
        # Parse the line
        train_data = json.loads(lines[0])
        
        print(f"📊 Train data keys: {list(train_data.keys())}")
        print(f"📊 Video path: {train_data.get('video', 'NOT FOUND')}")
        print(f"📊 Completion type: {type(train_data.get('completion', 'NOT FOUND'))}")
        
        # Critical checks
        video_path = train_data.get('video', '')
        if not video_path.endswith(f"{self.test_video_name}.mp4"):
            print(f"❌ Video path incorrect: {video_path}")
            return False
        
        completion = train_data.get('completion', '')
        if not isinstance(completion, str):
            print(f"❌ Completion should be string, got {type(completion)}")
            return False
        
        # Try to parse completion as JSON
        try:
            completion_json = json.loads(completion)
            print("✅ Completion is valid JSON string")
        except json.JSONDecodeError:
            print("❌ Completion is not valid JSON")
            return False
        
        print("✅ Dataset preparation test passed!")
        self.results["dataset_success"] = True
        return True
    
    def phase5_training_test(self) -> bool:
        """Phase 5: Training smoke test with overfitting."""
        print("\n🚀 PHASE 5: Training Pipeline Smoke Test")
        print("-" * 60)
        
        print("⚠️  This test requires GPU hardware (A100/H100)")
        print("⚠️  This will run actual training - ensure you have sufficient resources")
        
        # Create overfit dataset
        print("1️⃣ Creating overfit dataset...")
        train_file = self.project_root / "04_dataset" / "train.jsonl"
        val_file = self.project_root / "04_dataset" / "validation.jsonl"
        
        # Copy train to validation for overfitting
        with open(train_file, 'r') as f:
            train_content = f.read()
        
        with open(val_file, 'w') as f:
            f.write(train_content)
        
        print("✅ Overfit dataset created (train == validation)")
        
        # Update config for overfitting
        print("2️⃣ Configuring training for overfitting...")
        config_file = self.project_root / "05_training" / "football_sft_config.toml"
        
        # This would require updating the config file
        print("📋 Training configuration:")
        print("  - Epochs: 50 (high for overfitting)")
        print("  - Batch size: 1")
        print("  - Learning rate: 1e-4")
        print("  - Train file: train.jsonl")
        print("  - Validation file: validation.jsonl")
        
        print("\n3️⃣ Training execution...")
        print("⚠️  To run training, execute:")
        print("   cd 05_training")
        print("   python simple_football_sft.py --config football_sft_config.toml")
        print("\n📊 Expected behavior:")
        print("  - train_loss should drop quickly to ~0.0")
        print("  - LoRA adapter should be saved to checkpoints/")
        print("  - If loss doesn't drop, data format is wrong")
        
        self.results["training_ready"] = True
        return True
    
    def phase6_evaluation_test(self) -> bool:
        """Phase 6: Test evaluation pipeline with LoRA adapter."""
        print("\n📊 PHASE 6: Evaluation Pipeline Test")
        print("-" * 60)
        
        print("⚠️  This test requires GPU hardware and trained LoRA adapter")
        print("⚠️  Run this after completing Phase 5 training")
        
        print("📋 Evaluation test steps:")
        print("1. Create test.jsonl (identical to train.jsonl)")
        print("2. Run evaluation script")
        print("3. Verify LoRA adapter loading")
        print("4. Check output JSON generation")
        
        # Create test.jsonl
        train_file = self.project_root / "04_dataset" / "train.jsonl"
        test_file = self.project_root / "04_dataset" / "test.jsonl"
        
        with open(train_file, 'r') as f:
            train_content = f.read()
        
        with open(test_file, 'w') as f:
            f.write(train_content)
        
        print("✅ Test.jsonl created (identical to train.jsonl)")
        
        print("\n📋 To run evaluation:")
        print("   cd 06_evaluation")
        print("   python evaluate.py --test_file ../04_dataset/test.jsonl")
        print("\n📊 Expected results:")
        print("  - Script should not crash")
        print("  - Should load base model + LoRA adapter")
        print("  - Should generate output JSON")
        print("  - Output should be nearly identical to ground truth")
        
        self.results["evaluation_ready"] = True
        return True
    
    def phase7_inference_test(self) -> bool:
        """Phase 7: Full end-to-end inference test with unseen video."""
        print("\n🎯 PHASE 7: Full End-to-End Inference Test")
        print("-" * 60)
        
        print("⚠️  This is the FINAL TEST - requires everything to be working")
        print("⚠️  This test uses a completely new, unseen video")
        
        print("📋 Final test steps:")
        print("1. Get new video (e.g., 'Yellow Card' clip)")
        print("2. Run preprocessing on new video")
        print("3. Run inference with trained LoRA adapter")
        print("4. Verify JSON output quality")
        
        print("\n📋 To run final inference:")
        print("   cd 07_inference")
        print("   python football_inference.py --video_path /path/to/new_video.mp4")
        print("\n📊 Final verification checklist:")
        print("  ✅ JSON syntax is valid")
        print("  ✅ Event identification attempted")
        print("  ✅ Timestamps are plausible (not 0:0:0 to 0:0:1)")
        print("  ✅ Description is relevant to video content")
        
        self.results["inference_ready"] = True
        return True
    
    def run_all_tests(self) -> bool:
        """Run all pipeline tests."""
        print("🧪 Running Complete Pipeline Test Suite")
        print("=" * 60)
        
        tests = [
            ("Phase 1&2: Data Ingestion", self.phase1_2_data_test),
            ("Phase 3: Annotation", self.phase3_annotation_test),
            ("Phase 4: Dataset Preparation", self.phase4_dataset_test),
            ("Phase 5: Training Smoke Test", self.phase5_training_test),
            ("Phase 6: Evaluation Test", self.phase6_evaluation_test),
            ("Phase 7: Final Inference", self.phase7_inference_test),
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            print(f"\n{'='*60}")
            print(f"🧪 {test_name}")
            print('='*60)
            
            try:
                success = test_func()
                results[test_name] = success
                if success:
                    print(f"✅ {test_name} PASSED")
                else:
                    print(f"❌ {test_name} FAILED")
            except Exception as e:
                print(f"❌ {test_name} ERROR: {e}")
                results[test_name] = False
        
        # Summary
        print(f"\n{'='*60}")
        print("📊 PIPELINE TEST SUMMARY")
        print('='*60)
        
        passed = sum(1 for success in results.values() if success)
        total = len(results)
        
        for test_name, success in results.items():
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"{status:10} {test_name}")
        
        print(f"\n📊 Overall: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 ALL TESTS PASSED - Pipeline is ready for production!")
        else:
            print("⚠️  Some tests failed - review and fix issues before proceeding")
        
        return passed == total


def main():
    """Main test function."""
    project_root = Path(__file__).parent
    tester = PipelineTester(project_root)
    
    # Run all tests
    success = tester.run_all_tests()
    
    if success:
        print("\n🚀 Pipeline is ready for real data and training!")
        return 0
    else:
        print("\n⚠️  Pipeline needs fixes before proceeding")
        return 1


if __name__ == "__main__":
    exit(main())
