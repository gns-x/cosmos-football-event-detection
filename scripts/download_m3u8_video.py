#!/usr/bin/env python3
"""
M3U8 Stream Download Script
Downloads football videos from M3U8 streams using ffmpeg
Supports AWS MediaStore and other M3U8 sources
"""

import os
import subprocess
import sys
from pathlib import Path
import argparse
import time


def download_m3u8_stream(m3u8_url: str, output_path: str, duration: int = 30) -> bool:
    """Download video from M3U8 stream using ffmpeg."""
    print(f"📹 Downloading M3U8 stream from: {m3u8_url}")
    print(f"📁 Output path: {output_path}")
    print(f"⏱️  Duration: {duration} seconds")
    
    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Download M3U8 stream using ffmpeg
        cmd = [
            "ffmpeg",
            "-i", m3u8_url,
            "-t", str(duration),  # Limit duration
            "-c", "copy",  # Copy streams without re-encoding
            "-bsf:a", "aac_adtstoasc",  # Fix AAC audio
            "-y",  # Overwrite output file
            output_path
        ]
        
        print(f"📋 Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        if Path(output_path).exists():
            print(f"✅ M3U8 stream downloaded successfully: {output_path}")
            return True
        else:
            print(f"❌ Video file not found after download: {output_path}")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ M3U8 download failed: {e}")
        print(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ M3U8 download error: {e}")
        return False


def download_m3u8_with_ytdlp(m3u8_url: str, output_path: str) -> bool:
    """Download M3U8 stream using yt-dlp as fallback."""
    print(f"📹 Downloading M3U8 stream with yt-dlp: {m3u8_url}")
    
    try:
        cmd = [
            "yt-dlp",
            "-o", output_path,
            "--format", "best",
            "--no-playlist",
            m3u8_url
        ]
        
        print(f"📋 Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        if Path(output_path).exists():
            print(f"✅ M3U8 stream downloaded with yt-dlp: {output_path}")
            return True
        else:
            print(f"❌ Video file not found after yt-dlp download: {output_path}")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ yt-dlp M3U8 download failed: {e}")
        print(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ yt-dlp M3U8 download error: {e}")
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


def test_m3u8_url(m3u8_url: str) -> bool:
    """Test if M3U8 URL is accessible."""
    try:
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            m3u8_url
        ]
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=10)
        duration = float(result.stdout.strip())
        print(f"✅ M3U8 stream accessible, duration: {duration:.2f} seconds")
        return True
        
    except Exception as e:
        print(f"❌ M3U8 stream not accessible: {e}")
        return False


def main():
    """Main download function."""
    parser = argparse.ArgumentParser(description="Download football video from M3U8 stream")
    parser.add_argument("--url", type=str, 
                       default="https://h2ddsukdqkoh6b.data.mediastore.us-east-1.amazonaws.com/ucl-walionpapl-aug13-eb643a9-archive/master/ucl-walionpapl-aug13-eb643a9-archive.m3u8",
                       help="M3U8 stream URL")
    parser.add_argument("--output", type=str,
                       default="01_data_collection/raw_videos/goal_test_m3u8.mp4",
                       help="Output file path")
    parser.add_argument("--duration", type=int, default=30,
                       help="Duration to download in seconds")
    parser.add_argument("--method", type=str, choices=["ffmpeg", "yt-dlp", "auto"], default="auto",
                       help="Download method")
    
    args = parser.parse_args()
    
    print("🏈 M3U8 Football Video Download")
    print("=" * 50)
    print(f"🔗 M3U8 URL: {args.url}")
    print(f"📁 Output: {args.output}")
    print(f"⏱️  Duration: {args.duration} seconds")
    print(f"🔧 Method: {args.method}")
    print("")
    
    # Test M3U8 URL accessibility
    print("🔍 Testing M3U8 stream accessibility...")
    if not test_m3u8_url(args.url):
        print("❌ M3U8 stream is not accessible!")
        return 1
    
    # Download video
    success = False
    
    if args.method == "ffmpeg" or args.method == "auto":
        print("\n📹 Attempting download with ffmpeg...")
        success = download_m3u8_stream(args.url, args.output, args.duration)
    
    if not success and (args.method == "yt-dlp" or args.method == "auto"):
        print("\n📹 Attempting download with yt-dlp...")
        success = download_m3u8_with_ytdlp(args.url, args.output)
    
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
        
        print(f"\n✅ M3U8 video ready: {args.output}")
        print("🚀 Ready for Phase 2 preprocessing test!")
        return 0
    else:
        print("❌ M3U8 video download failed!")
        return 1


if __name__ == "__main__":
    exit(main())
