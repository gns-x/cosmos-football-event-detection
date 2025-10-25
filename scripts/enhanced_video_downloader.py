#!/usr/bin/env python3
"""
Enhanced Football Video Downloader
Uses multiple open source tools for comprehensive video downloading
"""

import os
import subprocess
import sys
from pathlib import Path
import argparse
import json
import time
from typing import List, Dict, Optional


class EnhancedVideoDownloader:
    """Enhanced video downloader using multiple open source tools."""
    
    def __init__(self, download_dir: str = "01_data_collection/raw_videos"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
        # Open source football channels
        self.football_channels = {
            "FIFA": "https://www.youtube.com/@FIFA",
            "UEFA": "https://www.youtube.com/@UEFA", 
            "PremierLeague": "https://www.youtube.com/@PremierLeague",
            "LaLiga": "https://www.youtube.com/@LaLiga",
            "SerieA": "https://www.youtube.com/@SerieA",
            "Bundesliga": "https://www.youtube.com/@Bundesliga",
            "Ligue1": "https://www.youtube.com/@Ligue1",
            "ChampionsLeague": "https://www.youtube.com/@ChampionsLeague",
            "EuropaLeague": "https://www.youtube.com/@EuropaLeague",
            "WorldCup": "https://www.youtube.com/@WorldCup"
        }
        
        # Football search terms
        self.football_searches = {
            "penalty_shot": "football penalty kick goal",
            "goal": "football goal scoring",
            "red_card": "football red card referee",
            "yellow_card": "football yellow card referee",
            "corner_kick": "football corner kick",
            "free_kick": "football free kick",
            "throw_in": "football throw in",
            "offside": "football offside VAR"
        }
        
        # M3U8 stream sources
        self.m3u8_streams = [
            "https://h2ddsukdqkoh6b.data.mediastore.us-east-1.amazonaws.com/ucl-walionpapl-aug13-eb643a9-archive/master/ucl-walionpapl-aug13-eb643a9-archive.m3u8"
            # Add more M3U8 streams here
        ]
    
    def download_with_ytdlp(self, url: str, output_dir: Path, max_downloads: int = 10, 
                          quality: str = "best[height<=720]") -> bool:
        """Download videos using yt-dlp."""
        try:
            print(f"📹 Downloading with yt-dlp: {url}")
            
            cmd = [
                "yt-dlp",
                "--max-downloads", str(max_downloads),
                "--format", quality,
                "--match-filter", "duration > 10 & duration < 60",
                "--output", str(output_dir / "%(title)s.%(ext)s"),
                "--write-info-json",
                "--write-thumbnail",
                "--extract-flat",
                url
            ]
            
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"✅ yt-dlp download completed: {output_dir}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ yt-dlp download failed: {e}")
            print(f"Error: {e.stderr}")
            return False
        except Exception as e:
            print(f"❌ yt-dlp error: {e}")
            return False
    
    def download_m3u8_stream(self, m3u8_url: str, output_path: Path, duration: int = 30) -> bool:
        """Download M3U8 stream using ffmpeg."""
        try:
            print(f"📡 Downloading M3U8 stream: {m3u8_url}")
            
            cmd = [
                "ffmpeg",
                "-i", m3u8_url,
                "-t", str(duration),
                "-c", "copy",
                "-bsf:a", "aac_adtstoasc",
                "-y",
                str(output_path)
            ]
            
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"✅ M3U8 stream downloaded: {output_path}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ M3U8 download failed: {e}")
            print(f"Error: {e.stderr}")
            return False
        except Exception as e:
            print(f"❌ M3U8 error: {e}")
            return False
    
    def download_from_channels(self, max_videos_per_channel: int = 10) -> Dict[str, int]:
        """Download videos from football channels."""
        print("📺 Downloading from open source football channels...")
        
        results = {}
        for channel_name, channel_url in self.football_channels.items():
            print(f"\n📡 Downloading from: {channel_name}")
            channel_dir = self.download_dir / "channels" / channel_name
            channel_dir.mkdir(parents=True, exist_ok=True)
            
            success = self.download_with_ytdlp(channel_url, channel_dir, max_videos_per_channel)
            if success:
                video_count = len(list(channel_dir.glob("*.mp4")))
                results[channel_name] = video_count
                print(f"✅ {channel_name}: {video_count} videos")
            else:
                results[channel_name] = 0
                print(f"❌ {channel_name}: 0 videos")
        
        return results
    
    def download_by_search_terms(self, max_videos_per_class: int = 5) -> Dict[str, int]:
        """Download videos by football search terms."""
        print("\n🔍 Downloading by football search terms...")
        
        results = {}
        for class_name, search_terms in self.football_searches.items():
            print(f"\n🎯 Downloading: {class_name}")
            class_dir = self.download_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            
            search_url = f"ytsearch20:{search_terms}"
            success = self.download_with_ytdlp(search_url, class_dir, max_videos_per_class)
            
            if success:
                video_count = len(list(class_dir.glob("*.mp4")))
                results[class_name] = video_count
                print(f"✅ {class_name}: {video_count} videos")
            else:
                results[class_name] = 0
                print(f"❌ {class_name}: 0 videos")
        
        return results
    
    def download_m3u8_streams(self, duration: int = 30) -> Dict[str, int]:
        """Download from M3U8 streams."""
        print("\n📡 Downloading M3U8 streams...")
        
        results = {}
        m3u8_dir = self.download_dir / "m3u8_streams"
        m3u8_dir.mkdir(parents=True, exist_ok=True)
        
        for i, stream_url in enumerate(self.m3u8_streams):
            print(f"\n📺 Downloading M3U8 stream {i+1}: {stream_url}")
            output_file = m3u8_dir / f"stream_{int(time.time())}_{i}.mp4"
            
            success = self.download_m3u8_stream(stream_url, output_file, duration)
            if success:
                results[f"stream_{i}"] = 1
                print(f"✅ M3U8 stream {i+1}: 1 video")
            else:
                results[f"stream_{i}"] = 0
                print(f"❌ M3U8 stream {i+1}: 0 videos")
        
        return results
    
    def get_download_summary(self) -> Dict[str, int]:
        """Get summary of downloaded videos."""
        summary = {}
        
        # Count videos in each directory
        for item in self.download_dir.rglob("*.mp4"):
            if item.is_file():
                relative_path = item.relative_to(self.download_dir)
                parent_dir = relative_path.parts[0]
                summary[parent_dir] = summary.get(parent_dir, 0) + 1
        
        return summary
    
    def run_comprehensive_download(self, max_videos_per_channel: int = 10, 
                                 max_videos_per_class: int = 5, 
                                 m3u8_duration: int = 30) -> Dict[str, any]:
        """Run comprehensive video download using all methods."""
        print("🚀 Enhanced Football Video Download")
        print("=" * 50)
        print(f"📁 Download directory: {self.download_dir}")
        print(f"🎯 Max videos per channel: {max_videos_per_channel}")
        print(f"🎯 Max videos per class: {max_videos_per_class}")
        print(f"⏱️  M3U8 duration: {m3u8_duration} seconds")
        print("")
        
        results = {}
        
        # Download from channels
        print("📺 Phase 1: Downloading from channels...")
        channel_results = self.download_from_channels(max_videos_per_channel)
        results["channels"] = channel_results
        
        # Download by search terms
        print("\n🔍 Phase 2: Downloading by search terms...")
        search_results = self.download_by_search_terms(max_videos_per_class)
        results["search_terms"] = search_results
        
        # Download M3U8 streams
        print("\n📡 Phase 3: Downloading M3U8 streams...")
        m3u8_results = self.download_m3u8_streams(m3u8_duration)
        results["m3u8_streams"] = m3u8_results
        
        # Get final summary
        print("\n📊 Final Summary:")
        final_summary = self.get_download_summary()
        results["final_summary"] = final_summary
        
        total_videos = sum(final_summary.values())
        print(f"  Total videos downloaded: {total_videos}")
        
        for category, count in final_summary.items():
            print(f"  {category}: {count} videos")
        
        print(f"\n✅ Enhanced download completed!")
        print(f"📁 Videos saved to: {self.download_dir}")
        
        return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Enhanced Football Video Downloader")
    parser.add_argument("--download_dir", type=str, default="01_data_collection/raw_videos",
                       help="Download directory")
    parser.add_argument("--max_videos_per_channel", type=int, default=10,
                       help="Max videos per channel")
    parser.add_argument("--max_videos_per_class", type=int, default=5,
                       help="Max videos per class")
    parser.add_argument("--m3u8_duration", type=int, default=30,
                       help="M3U8 stream duration")
    parser.add_argument("--output_json", type=str, default="download_results.json",
                       help="Output JSON file for results")
    
    args = parser.parse_args()
    
    # Create downloader
    downloader = EnhancedVideoDownloader(args.download_dir)
    
    # Run comprehensive download
    results = downloader.run_comprehensive_download(
        max_videos_per_channel=args.max_videos_per_channel,
        max_videos_per_class=args.max_videos_per_class,
        m3u8_duration=args.m3u8_duration
    )
    
    # Save results to JSON
    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Results saved to: {args.output_json}")
    return 0


if __name__ == "__main__":
    exit(main())
