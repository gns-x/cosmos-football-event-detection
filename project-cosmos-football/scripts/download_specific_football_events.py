#!/usr/bin/env python3
"""
Specific Football Event Video Downloader
Downloads videos for the 8 specific football event classes required for training
"""

import os
import subprocess
import sys
from pathlib import Path
import argparse
import json
import time
from typing import List, Dict, Optional

class SpecificFootballEventDownloader:
    """Downloads videos for specific football event classes."""
    
    def __init__(self, download_dir: str = "01_data_collection/raw_videos"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
        # The 8 specific event classes from Task.md
        self.event_classes = {
            "penalty_shot": {
                "search_terms": [
                    "football penalty kick compilation",
                    "penalty shootout moments",
                    "penalty kick goals saves",
                    "penalty kick highlights",
                    "penalty kick misses saves"
                ],
                "duration_range": "90-180"  # 90 seconds to 3 minutes
            },
            "goal": {
                "search_terms": [
                    "football goal compilation",
                    "amazing goals compilation",
                    "best goals of the season",
                    "football goal highlights",
                    "soccer goal moments"
                ],
                "duration_range": "90-180"
            },
            "goal_line_event": {
                "search_terms": [
                    "goal line technology moments",
                    "football goal line decisions",
                    "goal line controversy",
                    "goal line clearance",
                    "goal line technology VAR"
                ],
                "duration_range": "90-180"
            },
            "woodworks": {
                "search_terms": [
                    "football crossbar hits",
                    "goalpost hits compilation",
                    "woodwork moments football",
                    "crossbar challenge",
                    "unlucky shots crossbar"
                ],
                "duration_range": "90-180"
            },
            "shot_on_target": {
                "search_terms": [
                    "goalkeeper saves compilation",
                    "shot on target saves",
                    "amazing saves football",
                    "goalkeeper saves highlights",
                    "shot saved by goalkeeper"
                ],
                "duration_range": "90-180"
            },
            "red_card": {
                "search_terms": [
                    "football red card moments",
                    "red card incidents compilation",
                    "football red card fouls",
                    "red card referee decisions",
                    "football red card controversies"
                ],
                "duration_range": "90-180"
            },
            "yellow_card": {
                "search_terms": [
                    "football yellow card moments",
                    "yellow card compilation",
                    "football yellow card fouls",
                    "yellow card referee decisions",
                    "football yellow card incidents"
                ],
                "duration_range": "90-180"
            },
            "hat_trick": {
                "search_terms": [
                    "football hat trick moments",
                    "hat trick goals compilation",
                    "player hat trick highlights",
                    "hat trick third goal",
                    "football hat trick celebrations"
                ],
                "duration_range": "90-180"
            }
        }
    
    def download_videos_for_event(self, event_name: str, event_info: Dict, max_videos: int = 8) -> List[Path]:
        """Download videos for a specific event class."""
        print(f"🎯 Downloading {event_name.replace('_', ' ').title()} videos...")
        
        event_dir = self.download_dir / event_name
        event_dir.mkdir(exist_ok=True)
        
        downloaded_videos = []
        search_terms = event_info["search_terms"]
        duration_range = event_info["duration_range"]
        
        for i, search_term in enumerate(search_terms):
            if len(downloaded_videos) >= max_videos:
                break
                
            print(f"  🔍 Search {i+1}/{len(search_terms)}: {search_term}")
            
            try:
                # Download videos using yt-dlp with proper duration filter
                cmd = [
                    "yt-dlp",
                    "--max-downloads", str(max_videos - len(downloaded_videos)),
                    "--format", "best[height<=720]",
                    "--match-filter", f"duration > 90 & duration < 180",  # 90 seconds to 3 minutes
                    "--output", str(event_dir / f"{event_name}_%(title)s.%(ext)s"),
                    "--write-info-json",
                    "--write-thumbnail",
                    f"ytsearch10:{search_term}"
                ]
                
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                
                # Find downloaded videos
                for video_file in event_dir.glob(f"{event_name}_*.mp4"):
                    if video_file.is_file() and video_file not in downloaded_videos:
                        downloaded_videos.append(video_file)
                        print(f"    ✅ Downloaded: {video_file.name}")
                
                # Add delay between searches to avoid rate limiting
                time.sleep(2)
                
            except subprocess.CalledProcessError as e:
                print(f"    ❌ Download failed: {e}")
                continue
            except Exception as e:
                print(f"    ❌ Error: {e}")
                continue
        
        return downloaded_videos
    
    def download_m3u8_stream(self, m3u8_url: str, output_path: Path, duration: int = 120) -> Optional[Path]:
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
            
            if output_path.exists():
                print(f"✅ M3U8 stream downloaded: {output_path.name}")
                return output_path
            else:
                print(f"❌ M3U8 stream file not found: {output_path}")
                return None
                
        except subprocess.CalledProcessError as e:
            print(f"❌ M3U8 download failed: {e}")
            return None
        except Exception as e:
            print(f"❌ M3U8 error: {e}")
            return None
    
    def download_all_events(self, max_videos_per_event: int = 8) -> Dict[str, List[Path]]:
        """Download videos for all 8 event classes."""
        print("🚀 Downloading Specific Football Event Videos")
        print("=" * 60)
        print("📋 Required Event Classes:")
        for i, event_name in enumerate(self.event_classes.keys(), 1):
            print(f"  {i}. {event_name.replace('_', ' ').title()}")
        print("=" * 60)
        
        results = {}
        
        for event_name, event_info in self.event_classes.items():
            print(f"\n📁 Processing: {event_name.replace('_', ' ').title()}")
            
            # Download videos for this event
            downloaded_videos = self.download_videos_for_event(event_name, event_info, max_videos_per_event)
            results[event_name] = downloaded_videos
            
            print(f"  ✅ {event_name.replace('_', ' ').title()}: {len(downloaded_videos)} videos downloaded")
        
        return results
    
    def download_m3u8_content(self) -> List[Path]:
        """Download content from the provided M3U8 stream."""
        print("\n📡 Downloading M3U8 Stream Content...")
        
        m3u8_url = "https://h2ddsukdqkoh6b.data.mediastore.us-east-1.amazonaws.com/ucl-walionpapl-aug13-eb643a9-archive/master/ucl-walionpapl-aug13-eb643a9-archive.m3u8"
        
        # Create M3U8 directory
        m3u8_dir = self.download_dir / "m3u8_content"
        m3u8_dir.mkdir(exist_ok=True)
        
        downloaded_videos = []
        
        # Download multiple segments from the stream
        for i in range(3):  # Download 3 segments
            output_file = m3u8_dir / f"ucl_stream_segment_{i+1}.mp4"
            video = self.download_m3u8_stream(m3u8_url, output_file, 120)  # 2 minutes each
            if video:
                downloaded_videos.append(video)
        
        return downloaded_videos
    
    def validate_video_duration(self, video_path: Path) -> bool:
        """Validate that video is within required duration range."""
        try:
            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                str(video_path)
            ]
            
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            video_info = json.loads(result.stdout)
            
            duration = float(video_info.get("format", {}).get("duration", 0))
            
            # Check if duration is between 90 seconds and 3 minutes
            if 90 <= duration <= 180:
                return True
            else:
                print(f"    ⚠️  Duration {duration:.1f}s outside range (90-180s)")
                return False
                
        except Exception as e:
            print(f"    ❌ Error validating duration: {e}")
            return False
    
    def get_summary(self) -> Dict[str, int]:
        """Get summary of all downloaded videos."""
        summary = {}
        
        for event_dir in self.download_dir.iterdir():
            if event_dir.is_dir():
                event_name = event_dir.name
                video_count = len(list(event_dir.glob("*.mp4")))
                summary[event_name] = video_count
        
        return summary
    
    def create_dataset_metadata(self) -> Dict[str, any]:
        """Create metadata for the dataset."""
        metadata = {
            "dataset_info": {
                "name": "Football Event Classification Dataset",
                "description": "Videos for 8 specific football event classes",
                "total_classes": 8,
                "created_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "duration_requirement": "90-180 seconds"
            },
            "event_classes": list(self.event_classes.keys()),
            "video_counts": self.get_summary(),
            "data_format": {
                "input": "video file path",
                "output": "JSON with moment description, timestamps, and event type"
            }
        }
        
        return metadata

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Download Specific Football Event Videos")
    parser.add_argument("--download_dir", type=str, default="01_data_collection/raw_videos",
                       help="Download directory")
    parser.add_argument("--max_videos_per_event", type=int, default=8,
                       help="Max videos per event class")
    parser.add_argument("--include_m3u8", action="store_true",
                       help="Include M3U8 stream content")
    parser.add_argument("--output_metadata", type=str, default="dataset_metadata.json",
                       help="Output metadata file")
    
    args = parser.parse_args()
    
    # Create downloader
    downloader = SpecificFootballEventDownloader(args.download_dir)
    
    # Download videos for all events
    results = downloader.download_all_events(args.max_videos_per_event)
    
    # Download M3U8 content if requested
    if args.include_m3u8:
        m3u8_videos = downloader.download_m3u8_content()
        results["m3u8_content"] = m3u8_videos
    
    # Create metadata
    metadata = downloader.create_dataset_metadata()
    
    # Save metadata
    with open(args.output_metadata, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Print summary
    print("\n📊 Final Summary:")
    summary = downloader.get_summary()
    total_videos = sum(summary.values())
    
    for event_name, count in summary.items():
        print(f"  {event_name.replace('_', ' ').title()}: {count} videos")
    
    print(f"\n  Total videos: {total_videos}")
    print(f"📄 Metadata saved to: {args.output_metadata}")
    print("✅ Specific football event video download completed!")
    
    return 0

if __name__ == "__main__":
    exit(main())
