#!/usr/bin/env python3
"""
Test script to verify the annotation app works with the new video structure
"""

import requests
import json
from pathlib import Path

def test_annotation_app():
    """Test the annotation app endpoints."""
    base_url = "http://localhost:5000"
    
    print("🧪 Testing Annotation App")
    print("=" * 40)
    
    try:
        # Test event classes endpoint
        print("1. Testing event classes endpoint...")
        response = requests.get(f"{base_url}/api/event-classes")
        if response.status_code == 200:
            classes = response.json()
            print(f"   ✅ Found {len(classes)} event classes:")
            for cls in classes:
                print(f"      - {cls}")
        else:
            print(f"   ❌ Failed: {response.status_code}")
            return False
        
        # Test videos endpoint
        print("\n2. Testing videos endpoint...")
        response = requests.get(f"{base_url}/api/videos")
        if response.status_code == 200:
            videos = response.json()
            print(f"   ✅ Found {len(videos)} videos")
            
            # Group by class
            by_class = {}
            for video in videos:
                cls = video['class']
                if cls not in by_class:
                    by_class[cls] = 0
                by_class[cls] += 1
            
            print("   📊 Videos by class:")
            for cls, count in by_class.items():
                print(f"      - {cls}: {count} videos")
        else:
            print(f"   ❌ Failed: {response.status_code}")
            return False
        
        # Test progress endpoint
        print("\n3. Testing progress endpoint...")
        response = requests.get(f"{base_url}/api/progress")
        if response.status_code == 200:
            progress = response.json()
            print(f"   ✅ Progress: {progress['annotated']}/{progress['total']} ({progress['progress']:.1f}%)")
        else:
            print(f"   ❌ Failed: {response.status_code}")
            return False
        
        # Test video stats endpoint
        print("\n4. Testing video stats endpoint...")
        response = requests.get(f"{base_url}/api/video-stats")
        if response.status_code == 200:
            stats = response.json()
            print("   ✅ Video statistics:")
            for cls, data in stats.items():
                print(f"      - {cls}: {data['count']} videos")
        else:
            print(f"   ❌ Failed: {response.status_code}")
            return False
        
        print("\n✅ All tests passed! The annotation app is working correctly.")
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to the app. Make sure it's running on localhost:5000")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def check_video_structure():
    """Check if the video structure matches requirements."""
    print("\n📁 Checking Video Structure")
    print("=" * 40)
    
    raw_videos_dir = Path("01_data_collection/raw_videos")
    
    required_classes = [
        "penalty_shot", "goal", "goal_line_event", "woodworks",
        "shot_on_target", "red_card", "yellow_card", "hat_trick"
    ]
    
    total_videos = 0
    
    for cls in required_classes:
        class_dir = raw_videos_dir / cls
        if class_dir.exists():
            videos = list(class_dir.glob("*.mp4"))
            count = len(videos)
            total_videos += count
            status = "✅" if count > 0 else "⚠️"
            print(f"{status} {cls}: {count} videos")
            
            if count > 0:
                # Check duration of first video
                try:
                    import subprocess
                    result = subprocess.run([
                        "ffprobe", "-v", "quiet", "-print_format", "json",
                        "-show_format", str(videos[0])
                    ], capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        info = json.loads(result.stdout)
                        duration = float(info.get("format", {}).get("duration", 0))
                        print(f"    📹 Sample duration: {duration:.1f}s")
                except:
                    pass
        else:
            print(f"❌ {cls}: Directory not found")
    
    print(f"\n📊 Total videos: {total_videos}")
    
    if total_videos >= 8:  # At least 1 per class
        print("✅ Video structure looks good!")
        return True
    else:
        print("⚠️  Consider downloading more videos for better training data")
        return False

if __name__ == "__main__":
    print("🚀 Football Video Annotation App Test")
    print("=" * 50)
    
    # Check video structure first
    structure_ok = check_video_structure()
    
    if structure_ok:
        print("\n" + "=" * 50)
        print("To test the app, run:")
        print("cd 03_annotation/annotation_tool")
        print("python app.py")
        print("\nThen run this test script again to verify the API endpoints.")
    else:
        print("\n❌ Please fix the video structure before testing the app.")
