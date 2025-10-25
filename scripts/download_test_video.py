#!/usr/bin/env python3
"""
Download Test Video Script
Downloads a real football video for end-to-end pipeline testing
"""

import os
import subprocess
import sys
from pathlib import Path
import argparse


def download_football_video(video_url: str, output_path: str) -> bool:
    """Download a football video using yt-dlp."""
    print(f"📹 Downloading football video from: {video_url}")
    print(f"📁 Output path: {output_path}")
    
    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Download video using yt-dlp
        cmd = [
            "yt-dlp",
            "-o", output_path,
            "--format", "best[height<=720]",  # Limit to 720p for efficiency
            "--no-playlist",  # Download only single video
            video_url
        ]
        
        print(f"📋 Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        if Path(output_path).exists():
            print(f"✅ Video downloaded successfully: {output_path}")
            return True
        else:
            print(f"❌ Video file not found after download: {output_path}")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Download failed: {e}")
        print(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ Download error: {e}")
        return False


def get_video_info(video_path: str) -> dict:
    """Get video information using ffprobe."""
    try:
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            video_path
        ]
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        import json
        return json.loads(result.stdout)
        
    except Exception as e:
        print(f"⚠️  Could not get video info: {e}")
        return {}


def main():
    """Main download function."""
    parser = argparse.ArgumentParser(description="Download test football video")
    parser.add_argument("--url", type=str, 
                       default="https://www.youtube.com/watch?v=dQw4w9WgXcQ",  # Replace with real goal video
                       help="YouTube URL to download")
    parser.add_argument("--output", type=str,
                       default="01_data_collection/raw_videos/goal_test_01.mp4",
                       help="Output file path")
    
    args = parser.parse_args()
    
    print("🏈 Football Video Download Test")
    print("=" * 50)
    
    # Download video
    success = download_football_video(args.url, args.output)
    
    if success:
        # Get video info
        print("\n📊 Video Information:")
        info = get_video_info(args.output)
        
        if info:
            format_info = info.get("format", {})
            duration = format_info.get("duration", "Unknown")
            size = format_info.get("size", "Unknown")
            
            print(f"  Duration: {duration} seconds")
            print(f"  File size: {size} bytes")
            
            # Check streams
            streams = info.get("streams", [])
            for stream in streams:
                if stream.get("codec_type") == "video":
                    width = stream.get("width", "Unknown")
                    height = stream.get("height", "Unknown")
                    fps = stream.get("r_frame_rate", "Unknown")
                    print(f"  Resolution: {width}x{height}")
                    print(f"  FPS: {fps}")
        
        print(f"\n✅ Test video ready: {args.output}")
        print("🚀 Ready for Phase 2 preprocessing test!")
        return 0
    else:
        print("❌ Video download failed!")
        return 1


if __name__ == "__main__":
    exit(main())
