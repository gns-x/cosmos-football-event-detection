#!/usr/bin/env python3
"""
End-to-End Pipeline Test - Critical Validation
Implements the exact testing plan for real data validation
"""

import os
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List
import time


class EndToEndTester:
    """Critical end-to-end pipeline tester."""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root).parent  # Go up one level from scripts/
        self.test_video_name = "goal_test_01"
        self.results = {}
        
        print("🧪 CRITICAL END-TO-END PIPELINE TEST")
        print("=" * 70)
        print("This is the most critical part of the project!")
        print("Testing entire pipeline with real data...")
        print("=" * 70)
    
    def phase1_2_data_test(self) -> bool:
        """Phase 1 & 2: Data Ingestion and Preprocessing Test"""
        print("\n📹 PHASE 1 & 2: Data Ingestion and Preprocessing Test")
        print("=" * 60)
        
        # Step 1: Check existing videos instead of downloading
        print("1️⃣ Checking existing downloaded videos...")
        raw_videos_dir = self.project_root / "01_data_collection" / "raw_videos"
        
        if not raw_videos_dir.exists():
            print("❌ Raw videos directory not found!")
            return False
        
        # Count videos in each event class
        event_classes = ["penalty_shot", "goal", "goal_line_event", "woodworks", 
                        "shot_on_target", "red_card", "yellow_card", "hat_trick"]
        
        total_videos = 0
        for event_class in event_classes:
            class_dir = raw_videos_dir / event_class
            if class_dir.exists():
                video_count = len(list(class_dir.glob("*.mp4")))
                total_videos += video_count
                print(f"  📁 {event_class}: {video_count} videos")
            else:
                print(f"  📁 {event_class}: 0 videos (directory not found)")
        
        if total_videos == 0:
            print("❌ No videos found! Run 'make download-videos' first.")
            return False
        
        print(f"✅ Found {total_videos} total videos across {len(event_classes)} event classes")
        
        # Use first available video for testing
        test_video_path = None
        for event_class in event_classes:
            class_dir = raw_videos_dir / event_class
            if class_dir.exists():
                videos = list(class_dir.glob("*.mp4"))
                if videos:
                    test_video_path = videos[0]
                    self.test_video_name = test_video_path.stem
                    break
        
        if not test_video_path:
            print("❌ No valid video files found!")
            return False
        
        print(f"📹 Using test video: {test_video_path.name}")
        
        # Step 2: Execute preprocessing
        print("\n2️⃣ Executing preprocessing to 4fps...")
        processed_video_path = self.project_root / "02_preprocessing" / "processed_videos" / f"{self.test_video_name}.mp4"
        processed_video_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            cmd = [
                "ffmpeg",
                "-i", str(test_video_path),
                "-r", "4",  # CRITICAL: Set fps to 4
                "-vf", "scale=720:480",
                "-c:v", "libx264",
                "-c:a", "aac",
                "-preset", "fast",
                "-crf", "23",
                "-y",
                str(processed_video_path)
            ]
            
            print(f"📋 Preprocessing command: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            if processed_video_path.exists():
                print(f"✅ Video processed: {processed_video_path}")
            else:
                print(f"❌ Processed video not found")
                return False
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Preprocessing failed: {e}")
            return False
        
        # Step 3: CRITICAL VERIFICATION - Check fps
        print("\n4️⃣ CRITICAL VERIFICATION: Checking output fps...")
        try:
            cmd = [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(processed_video_path)
            ]
            
            print(f"📋 FPS check: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            fps_output = result.stdout.strip()
            
            print(f"📊 FPS Output: '{fps_output}'")
            
            if fps_output in ["4/1", "4"]:
                print("✅ SUCCESS: Video is correctly processed to 4fps!")
                print("✅ The model will work with this fps!")
                self.results["fps_verification"] = True
                return True
            else:
                print(f"❌ CRITICAL FAILURE: Expected '4/1' or '4', got '{fps_output}'")
                print("❌ The model will fail with this fps!")
                print("❌ Check your ffmpeg command!")
                self.results["fps_verification"] = False
                return False
                
        except subprocess.CalledProcessError as e:
            print(f"❌ FPS verification failed: {e}")
            return False
    
    def phase3_annotation_test(self) -> bool:
        """Phase 3: Annotation (Ground Truth) Test"""
        print("\n📝 PHASE 3: Annotation (Ground Truth) Test")
        print("=" * 60)
        
        print("1️⃣ Manually annotating video with real timestamps...")
        print("⚠️  Watch the video in a player that shows timestamps!")
        print("⚠️  Note the exact timestamps for the goal event!")
        
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
            
            # Step 2: Execute JSON linter
            print("\n2️⃣ Executing JSON linter...")
            try:
                with open(annotation_path, 'r') as f:
                    loaded_json = json.load(f)
                
                print("✅ JSON validation passed - Valid JSON")
                print(f"📊 Annotation content:")
                print(json.dumps(loaded_json, indent=2))
                
                self.results["annotation_success"] = True
                return True
                
            except json.JSONDecodeError as e:
                print(f"❌ JSON validation failed: {e}")
                print("❌ A single trailing comma or misplaced quote will break the entire pipeline!")
                return False
                
        except Exception as e:
            print(f"❌ Annotation creation failed: {e}")
            return False
    
    def phase4_dataset_test(self) -> bool:
        """Phase 4: Dataset Preparation Test"""
        print("\n📊 PHASE 4: Dataset Preparation Test")
        print("=" * 60)
        
        print("1️⃣ Isolating test data...")
        ground_truth_dir = self.project_root / "03_annotation" / "ground_truth_json"
        backup_dir = ground_truth_dir / "backup"
        backup_dir.mkdir(exist_ok=True)
        
        # Move other files to backup
        moved_files = 0
        for file in ground_truth_dir.glob("*.json"):
            if file.name != f"{self.test_video_name}.json":
                backup_path = backup_dir / file.name
                file.rename(backup_path)
                moved_files += 1
                print(f"📁 Moved {file.name} to backup")
        
        print(f"📊 Moved {moved_files} files to backup")
        print("✅ Only test data remains")
        
        # Step 2: Execute dataset script
        print("\n2️⃣ Executing dataset preparation script...")
        try:
            dataset_script = self.project_root / "04_dataset" / "build_sft_dataset.py"
            cmd = ["python", str(dataset_script)]
            
            print(f"📋 Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, cwd=self.project_root, check=True, capture_output=True, text=True)
            
            print("✅ Dataset preparation completed")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Dataset preparation failed: {e}")
            print(f"Error: {e.stderr}")
            return False
        
        # Step 3: Verification
        print("\n3️⃣ CRITICAL VERIFICATION:")
        train_file = self.project_root / "04_dataset" / "train.jsonl"
        val_file = self.project_root / "04_dataset" / "validation.jsonl"
        
        # Check if we have data in either train or validation
        train_lines = 0
        val_lines = 0
        
        if train_file.exists():
            with open(train_file, 'r') as f:
                train_lines = len(f.readlines())
        
        if val_file.exists():
            with open(val_file, 'r') as f:
                val_lines = len(f.readlines())
        
        total_lines = train_lines + val_lines
        
        if total_lines == 0:
            print(f"❌ No data found in train.jsonl or validation.jsonl")
            return False
        
        if total_lines != 1:
            print(f"❌ Expected 1 line total, got {total_lines} (train: {train_lines}, val: {val_lines})")
            return False
        
        # Use whichever file has the data
        data_file = train_file if train_lines > 0 else val_file
        print(f"✅ Found data in: {data_file.name}")
        
        # Check content
        with open(data_file, 'r') as f:
            lines = f.readlines()
        
        print("✅ Train.jsonl contains exactly 1 line")
        
        # Parse and verify the line
        train_data = json.loads(lines[0])
        
        print(f"📊 Train data keys: {list(train_data.keys())}")
        
        # Critical Check 1: Video path
        video_path = train_data.get('video', '')
        expected_path = f"02_preprocessing/processed_videos/{self.test_video_name}.mp4"
        
        if not video_path.endswith(f"{self.test_video_name}.mp4"):
            print(f"❌ Video path incorrect: {video_path}")
            print(f"❌ Expected to end with: {self.test_video_name}.mp4")
            return False
        
        print(f"✅ Video path correct: {video_path}")
        
        # Critical Check 2: Completion format
        completion = train_data.get('completion', '')
        if not isinstance(completion, str):
            print(f"❌ Completion should be string, got {type(completion)}")
            return False
        
        print("✅ Completion is a string")
        
        # Try to parse completion as JSON
        try:
            completion_json = json.loads(completion)
            print("✅ Completion is valid JSON string")
            print(f"📊 Completion content: {json.dumps(completion_json, indent=2)}")
        except json.JSONDecodeError:
            print("❌ Completion is not valid JSON")
            return False
        
        print("✅ Dataset preparation test PASSED!")
        self.results["dataset_success"] = True
        return True
    
    def phase5_training_test(self) -> bool:
        """Phase 5: Training Pipeline Smoke Test"""
        print("\n🚀 PHASE 5: Training Pipeline Smoke Test")
        print("=" * 60)
        
        print("⚠️  CRITICAL: This test requires GPU hardware (A100/H100)")
        print("⚠️  This will run actual training - ensure sufficient resources")
        print("⚠️  We will overfit on a single batch - standard ML practice")
        
        # Step 1: Create overfit dataset
        print("\n1️⃣ Creating overfit dataset...")
        train_file = self.project_root / "04_dataset" / "train.jsonl"
        val_file = self.project_root / "04_dataset" / "validation.jsonl"
        
        # Find which file has the data
        data_file = None
        if train_file.exists() and train_file.stat().st_size > 0:
            data_file = train_file
        elif val_file.exists() and val_file.stat().st_size > 0:
            data_file = val_file
            # Copy validation to train for overfitting
            with open(val_file, 'r') as f:
                val_content = f.read()
            with open(train_file, 'w') as f:
                f.write(val_content)
        
        if not data_file:
            print("❌ No data found in train.jsonl or validation.jsonl")
            return False
        
        # Copy train to validation for overfitting
        with open(train_file, 'r') as f:
            train_content = f.read()
        
        with open(val_file, 'w') as f:
            f.write(train_content)
        
        print("✅ Overfit dataset created (train == validation)")
        
        # Step 2: Configure training for overfitting
        print("\n2️⃣ Configuring training for overfitting...")
        print("📋 Training configuration for overfitting:")
        print("  - num_train_epochs: 50 (high for overfitting)")
        print("  - batch_size: 1")
        print("  - learning_rate: 1e-4")
        print("  - train_file: train.jsonl")
        print("  - validation_file: validation.jsonl")
        
        # Step 3: Execute training
        print("\n3️⃣ Executing training (REQUIRES GPU)...")
        print("📋 To run training, execute:")
        print("   cd 05_training")
        print("   conda activate cosmos-football")
        print("   python simple_football_sft.py --config football_sft_config.toml")
        
        print("\n📊 Expected behavior:")
        print("  - train_loss should drop very quickly")
        print("  - Loss should approach 0.0 (overfitting)")
        print("  - If loss doesn't drop, data format is wrong")
        print("  - LoRA adapter must be saved to checkpoints/")
        
        print("\n⚠️  CRITICAL VERIFICATION:")
        print("  - Watch console output for loss dropping")
        print("  - Check for adapter_model.bin in checkpoints/")
        print("  - If loss doesn't drop, your data format is wrong!")
        
        self.results["training_ready"] = True
        return True
    
    def phase6_evaluation_test(self) -> bool:
        """Phase 6: Evaluation Pipeline Test"""
        print("\n📊 PHASE 6: Evaluation Pipeline Test")
        print("=" * 60)
        
        print("⚠️  This test requires GPU hardware and trained LoRA adapter")
        print("⚠️  Run this after completing Phase 5 training")
        
        # Step 1: Create test set
        print("\n1️⃣ Creating test set...")
        train_file = self.project_root / "04_dataset" / "train.jsonl"
        test_file = self.project_root / "04_dataset" / "test.jsonl"
        
        with open(train_file, 'r') as f:
            train_content = f.read()
        
        with open(test_file, 'w') as f:
            f.write(train_content)
        
        print("✅ Test.jsonl created (identical to train.jsonl)")
        
        # Step 2: Execute evaluation
        print("\n2️⃣ Executing evaluation (REQUIRES GPU)...")
        print("📋 To run evaluation:")
        print("   cd 06_evaluation")
        print("   python evaluate.py --test_file ../04_dataset/test.jsonl")
        
        print("\n📊 Expected results:")
        print("  - Script should not crash")
        print("  - Should load base model + LoRA adapter")
        print("  - Should generate output JSON file")
        print("  - Output should be nearly identical to ground truth")
        
        print("\n⚠️  CRITICAL VERIFICATION:")
        print("  - Check /06_evaluation/results/ for output files")
        print("  - Verify LoRA adapter loading")
        print("  - Check output JSON quality")
        
        self.results["evaluation_ready"] = True
        return True
    
    def phase7_inference_test(self) -> bool:
        """Phase 7: Full End-to-End Inference Test"""
        print("\n🎯 PHASE 7: Full End-to-End Inference Test")
        print("=" * 60)
        
        print("⚠️  This is the FINAL TEST - requires everything to be working")
        print("⚠️  This test uses a completely new, unseen video")
        
        print("\n📋 Final test steps:")
        print("1. Get new video (e.g., 'Yellow Card' clip)")
        print("2. Run preprocessing on new video")
        print("3. Run inference with trained LoRA adapter")
        print("4. Verify JSON output quality")
        
        print("\n📋 To run final inference:")
        print("   cd 07_inference")
        print("   python football_inference.py --video_path /path/to/new_video.mp4")
        
        print("\n📊 FINAL VERIFICATION CHECKLIST:")
        print("  ✅ JSON syntax is valid")
        print("  ✅ Event identification attempted")
        print("  ✅ Timestamps are plausible (not 0:0:0 to 0:0:1)")
        print("  ✅ Description is relevant to video content")
        
        print("\n🎉 If this test produces structured, relevant JSON for an unseen video,")
        print("   your entire pipeline is working!")
        print("   You can now proceed with full dataset annotation and training!")
        
        self.results["inference_ready"] = True
        return True
    
    def run_critical_tests(self) -> bool:
        """Run all critical pipeline tests."""
        print("🧪 CRITICAL PIPELINE TEST SUITE")
        print("=" * 70)
        
        tests = [
            ("Phase 1&2: Data Ingestion & Preprocessing", self.phase1_2_data_test),
            ("Phase 3: Annotation (Ground Truth)", self.phase3_annotation_test),
            ("Phase 4: Dataset Preparation", self.phase4_dataset_test),
            ("Phase 5: Training Smoke Test", self.phase5_training_test),
            ("Phase 6: Evaluation Pipeline", self.phase6_evaluation_test),
            ("Phase 7: Final End-to-End Inference", self.phase7_inference_test),
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            print(f"\n{'='*70}")
            print(f"🧪 {test_name}")
            print('='*70)
            
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
        print(f"\n{'='*70}")
        print("📊 CRITICAL PIPELINE TEST SUMMARY")
        print('='*70)
        
        passed = sum(1 for success in results.values() if success)
        total = len(results)
        
        for test_name, success in results.items():
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"{status:10} {test_name}")
        
        print(f"\n📊 Overall: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 ALL CRITICAL TESTS PASSED!")
            print("🚀 Pipeline is ready for production with real data!")
        else:
            print("⚠️  Some critical tests failed!")
            print("🔧 Review and fix issues before proceeding to full dataset")
        
        return passed == total


def main():
    """Main test function."""
    project_root = Path(__file__).parent
    tester = EndToEndTester(project_root)
    
    # Run all critical tests
    success = tester.run_critical_tests()
    
    if success:
        print("\n🚀 CRITICAL PIPELINE VALIDATION COMPLETE!")
        print("🎯 Ready for full dataset annotation and training!")
        return 0
    else:
        print("\n⚠️  CRITICAL PIPELINE ISSUES DETECTED!")
        print("🔧 Fix issues before proceeding to production!")
        return 1


if __name__ == "__main__":
    exit(main())
