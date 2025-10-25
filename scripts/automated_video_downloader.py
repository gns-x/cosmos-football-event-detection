#!/usr/bin/env python3
"""
Automated Football Video Downloader
Downloads videos from open sources automatically and validates them
"""

import os
import subprocess
import sys
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import argparse
from datetime import datetime
import hashlib


class AutomatedVideoDownloader:
    """Automated video downloader with validation."""
    
    def __init__(self, download_dir: str = "01_data_collection/raw_videos"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
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
        
        # M3U8 stream sources (expanded list)
        self.m3u8_streams = [
            # AWS MediaStore streams
            "https://h2ddsukdqkoh6b.data.mediastore.us-east-1.amazonaws.com/ucl-walionpapl-aug13-eb643a9-archive/master/ucl-walionpapl-aug13-eb643a9-archive.m3u8",
            
            # Archive.org streams
            "https://archive.org/download/football_highlights/football_highlights.m3u8",
            "https://archive.org/download/soccer_goals/soccer_goals.m3u8",
            "https://archive.org/download/football_matches/football_matches.m3u8",
            
            # Educational/Open source sports content
            "https://open-source-sports.example.com/football.m3u8",
            "https://educational-sports.example.com/soccer.m3u8",
            
            # Public domain sports content
            "https://public-sports.example.com/football.m3u8",
            "https://free-sports.example.com/soccer.m3u8",
            
            # IPTV sources (free channels)
            "https://iptv.example.com/sports.m3u8",
            "https://free-iptv.example.com/football.m3u8",
            
            # Additional discovered streams (will be populated by discovery script)
        ]
        
        # Load discovered streams if available
        self.load_discovered_streams()
        
        # Validation rules
        self.validation_rules = {
            "min_duration": 5,  # seconds
            "max_duration": 300,  # seconds
            "min_file_size": 100000,  # bytes (100KB)
            "max_file_size": 500000000,  # bytes (500MB)
            "required_codecs": ["h264", "avc1"],
            "min_resolution": (320, 240),
            "max_resolution": (1920, 1080)
        }
        
        self.download_stats = {
            "total_attempted": 0,
            "total_downloaded": 0,
            "total_validated": 0,
            "total_failed": 0,
            "start_time": datetime.now().isoformat()
        }
    
    def load_discovered_streams(self):
        """Load discovered M3U8 streams from discovery results."""
        try:
            discovery_file = Path("discovered_m3u8_streams.json")
            if discovery_file.exists():
                with open(discovery_file, 'r') as f:
                    discovery_data = json.load(f)
                
                # Add discovered streams to the list
                for stream in discovery_data.get("valid_streams", []):
                    if stream.get("url") not in self.m3u8_streams:
                        self.m3u8_streams.append(stream["url"])
                        self.logger.info(f"📡 Added discovered stream: {stream['url']}")
                
                self.logger.info(f"✅ Loaded {len(discovery_data.get('valid_streams', []))} discovered streams")
            else:
                self.logger.info("ℹ️  No discovered streams file found, using default streams")
        except Exception as e:
            self.logger.error(f"❌ Error loading discovered streams: {e}")
    
    def setup_logging(self):
        """Setup logging for the downloader."""
        log_dir = self.download_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"download_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def validate_video(self, video_path: Path) -> Tuple[bool, Dict[str, any]]:
        """Validate a video file using ffprobe."""
        try:
            self.logger.info(f"🔍 Validating video: {video_path.name}")
            
            # Get video information using ffprobe
            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                str(video_path)
            ]
            
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            video_info = json.loads(result.stdout)
            
            # Extract video details
            format_info = video_info.get("format", {})
            streams = video_info.get("streams", [])
            
            # Find video stream
            video_stream = None
            for stream in streams:
                if stream.get("codec_type") == "video":
                    video_stream = stream
                    break
            
            if not video_stream:
                return False, {"error": "No video stream found"}
            
            # Validation checks
            duration = float(format_info.get("duration", 0))
            file_size = int(format_info.get("size", 0))
            width = int(video_stream.get("width", 0))
            height = int(video_stream.get("height", 0))
            codec = video_stream.get("codec_name", "")
            
            validation_result = {
                "duration": duration,
                "file_size": file_size,
                "resolution": f"{width}x{height}",
                "codec": codec,
                "fps": video_stream.get("r_frame_rate", "unknown"),
                "bitrate": format_info.get("bit_rate", "unknown")
            }
            
            # Check duration
            if duration < self.validation_rules["min_duration"]:
                return False, {**validation_result, "error": f"Duration too short: {duration}s < {self.validation_rules['min_duration']}s"}
            
            if duration > self.validation_rules["max_duration"]:
                return False, {**validation_result, "error": f"Duration too long: {duration}s > {self.validation_rules['max_duration']}s"}
            
            # Check file size
            if file_size < self.validation_rules["min_file_size"]:
                return False, {**validation_result, "error": f"File too small: {file_size} bytes < {self.validation_rules['min_file_size']} bytes"}
            
            if file_size > self.validation_rules["max_file_size"]:
                return False, {**validation_result, "error": f"File too large: {file_size} bytes > {self.validation_rules['max_file_size']} bytes"}
            
            # Check resolution
            if width < self.validation_rules["min_resolution"][0] or height < self.validation_rules["min_resolution"][1]:
                return False, {**validation_result, "error": f"Resolution too low: {width}x{height} < {self.validation_rules['min_resolution'][0]}x{self.validation_rules['min_resolution'][1]}"}
            
            if width > self.validation_rules["max_resolution"][0] or height > self.validation_rules["max_resolution"][1]:
                return False, {**validation_result, "error": f"Resolution too high: {width}x{height} > {self.validation_rules['max_resolution'][0]}x{self.validation_rules['max_resolution'][1]}"}
            
            # Check codec
            if codec not in self.validation_rules["required_codecs"]:
                return False, {**validation_result, "error": f"Unsupported codec: {codec} not in {self.validation_rules['required_codecs']}"}
            
            self.logger.info(f"✅ Video validated: {video_path.name}")
            return True, validation_result
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"❌ FFprobe failed for {video_path.name}: {e}")
            return False, {"error": f"FFprobe failed: {e}"}
        except Exception as e:
            self.logger.error(f"❌ Validation error for {video_path.name}: {e}")
            return False, {"error": f"Validation error: {e}"}
    
    def download_with_ytdlp(self, url: str, output_dir: Path, max_downloads: int = 10, 
                          quality: str = "best[height<=720]") -> List[Path]:
        """Download videos using yt-dlp with validation."""
        downloaded_videos = []
        
        try:
            self.logger.info(f"📹 Downloading with yt-dlp: {url}")
            
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
            
            # Find downloaded videos
            for video_file in output_dir.glob("*.mp4"):
                if video_file.is_file():
                    downloaded_videos.append(video_file)
            
            self.logger.info(f"✅ yt-dlp download completed: {len(downloaded_videos)} videos")
            return downloaded_videos
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"❌ yt-dlp download failed: {e}")
            return []
        except Exception as e:
            self.logger.error(f"❌ yt-dlp error: {e}")
            return []
    
    def download_m3u8_stream(self, m3u8_url: str, output_path: Path, duration: int = 30) -> Optional[Path]:
        """Download M3U8 stream using ffmpeg with validation."""
        try:
            self.logger.info(f"📡 Downloading M3U8 stream: {m3u8_url}")
            
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
                self.logger.info(f"✅ M3U8 stream downloaded: {output_path.name}")
                return output_path
            else:
                self.logger.error(f"❌ M3U8 stream file not found: {output_path}")
                return None
                
        except subprocess.CalledProcessError as e:
            self.logger.error(f"❌ M3U8 download failed: {e}")
            return None
        except Exception as e:
            self.logger.error(f"❌ M3U8 error: {e}")
            return None
    
    def process_downloaded_videos(self, videos: List[Path], category: str) -> Dict[str, any]:
        """Process and validate downloaded videos."""
        results = {
            "category": category,
            "total_downloaded": len(videos),
            "valid_videos": [],
            "invalid_videos": [],
            "validation_errors": []
        }
        
        for video in videos:
            self.download_stats["total_attempted"] += 1
            
            # Validate video
            is_valid, validation_info = self.validate_video(video)
            
            if is_valid:
                self.download_stats["total_validated"] += 1
                results["valid_videos"].append({
                    "file": str(video),
                    "info": validation_info
                })
                self.logger.info(f"✅ Valid video: {video.name}")
            else:
                self.download_stats["total_failed"] += 1
                results["invalid_videos"].append({
                    "file": str(video),
                    "error": validation_info.get("error", "Unknown error")
                })
                results["validation_errors"].append(validation_info.get("error", "Unknown error"))
                self.logger.warning(f"❌ Invalid video: {video.name} - {validation_info.get('error', 'Unknown error')}")
                
                # Remove invalid video
                try:
                    video.unlink()
                    self.logger.info(f"🗑️  Removed invalid video: {video.name}")
                except Exception as e:
                    self.logger.error(f"❌ Failed to remove invalid video {video.name}: {e}")
        
        return results
    
    def download_from_channels(self, max_videos_per_channel: int = 10) -> Dict[str, any]:
        """Download videos from football channels with validation."""
        self.logger.info("📺 Downloading from open source football channels...")
        
        all_results = {}
        
        for channel_name, channel_url in self.football_channels.items():
            self.logger.info(f"📡 Downloading from: {channel_name}")
            channel_dir = self.download_dir / "channels" / channel_name
            channel_dir.mkdir(parents=True, exist_ok=True)
            
            # Download videos
            downloaded_videos = self.download_with_ytdlp(channel_url, channel_dir, max_videos_per_channel)
            
            # Process and validate
            if downloaded_videos:
                results = self.process_downloaded_videos(downloaded_videos, f"channel_{channel_name}")
                all_results[channel_name] = results
                self.logger.info(f"✅ {channel_name}: {results['total_downloaded']} downloaded, {len(results['valid_videos'])} valid")
            else:
                self.logger.warning(f"⚠️  {channel_name}: No videos downloaded")
                all_results[channel_name] = {
                    "category": f"channel_{channel_name}",
                    "total_downloaded": 0,
                    "valid_videos": [],
                    "invalid_videos": [],
                    "validation_errors": []
                }
        
        return all_results
    
    def download_by_search_terms(self, max_videos_per_class: int = 5) -> Dict[str, any]:
        """Download videos by football search terms with validation."""
        self.logger.info("🔍 Downloading by football search terms...")
        
        all_results = {}
        
        for class_name, search_terms in self.football_searches.items():
            self.logger.info(f"🎯 Downloading: {class_name}")
            class_dir = self.download_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            
            search_url = f"ytsearch20:{search_terms}"
            downloaded_videos = self.download_with_ytdlp(search_url, class_dir, max_videos_per_class)
            
            # Process and validate
            if downloaded_videos:
                results = self.process_downloaded_videos(downloaded_videos, f"search_{class_name}")
                all_results[class_name] = results
                self.logger.info(f"✅ {class_name}: {results['total_downloaded']} downloaded, {len(results['valid_videos'])} valid")
            else:
                self.logger.warning(f"⚠️  {class_name}: No videos downloaded")
                all_results[class_name] = {
                    "category": f"search_{class_name}",
                    "total_downloaded": 0,
                    "valid_videos": [],
                    "invalid_videos": [],
                    "validation_errors": []
                }
        
        return all_results
    
    def download_m3u8_streams(self, duration: int = 30) -> Dict[str, any]:
        """Download from M3U8 streams with validation."""
        self.logger.info("📡 Downloading M3U8 streams...")
        
        all_results = {}
        m3u8_dir = self.download_dir / "m3u8_streams"
        m3u8_dir.mkdir(parents=True, exist_ok=True)
        
        for i, stream_url in enumerate(self.m3u8_streams):
            self.logger.info(f"📺 Downloading M3U8 stream {i+1}: {stream_url}")
            output_file = m3u8_dir / f"stream_{int(time.time())}_{i}.mp4"
            
            downloaded_video = self.download_m3u8_stream(stream_url, output_file, duration)
            
            if downloaded_video:
                # Process and validate
                results = self.process_downloaded_videos([downloaded_video], f"m3u8_stream_{i}")
                all_results[f"stream_{i}"] = results
                self.logger.info(f"✅ M3U8 stream {i+1}: {results['total_downloaded']} downloaded, {len(results['valid_videos'])} valid")
            else:
                self.logger.warning(f"⚠️  M3U8 stream {i+1}: No video downloaded")
                all_results[f"stream_{i}"] = {
                    "category": f"m3u8_stream_{i}",
                    "total_downloaded": 0,
                    "valid_videos": [],
                    "invalid_videos": [],
                    "validation_errors": []
                }
        
        return all_results
    
    def run_automated_download(self, max_videos_per_channel: int = 10, 
                             max_videos_per_class: int = 5, 
                             m3u8_duration: int = 30) -> Dict[str, any]:
        """Run automated video download with validation."""
        self.logger.info("🚀 Automated Football Video Download")
        self.logger.info("=" * 50)
        self.logger.info(f"📁 Download directory: {self.download_dir}")
        self.logger.info(f"🎯 Max videos per channel: {max_videos_per_channel}")
        self.logger.info(f"🎯 Max videos per class: {max_videos_per_class}")
        self.logger.info(f"⏱️  M3U8 duration: {m3u8_duration} seconds")
        self.logger.info("")
        
        results = {}
        
        # Download from channels
        self.logger.info("📺 Phase 1: Downloading from channels...")
        channel_results = self.download_from_channels(max_videos_per_channel)
        results["channels"] = channel_results
        
        # Download by search terms
        self.logger.info("🔍 Phase 2: Downloading by search terms...")
        search_results = self.download_by_search_terms(max_videos_per_class)
        results["search_terms"] = search_results
        
        # Download M3U8 streams
        self.logger.info("📡 Phase 3: Downloading M3U8 streams...")
        m3u8_results = self.download_m3u8_streams(m3u8_duration)
        results["m3u8_streams"] = m3u8_results
        
        # Final summary
        self.download_stats["end_time"] = datetime.now().isoformat()
        results["download_stats"] = self.download_stats
        
        # Log final summary
        self.logger.info("📊 Final Summary:")
        self.logger.info(f"  Total attempted: {self.download_stats['total_attempted']}")
        self.logger.info(f"  Total validated: {self.download_stats['total_validated']}")
        self.logger.info(f"  Total failed: {self.download_stats['total_failed']}")
        
        # Count valid videos by category
        total_valid = 0
        for category, category_results in results.items():
            if isinstance(category_results, dict) and "valid_videos" in category_results:
                valid_count = len(category_results["valid_videos"])
                total_valid += valid_count
                self.logger.info(f"  {category}: {valid_count} valid videos")
        
        self.logger.info(f"  Total valid videos: {total_valid}")
        self.logger.info(f"✅ Automated download completed!")
        
        return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Automated Football Video Downloader")
    parser.add_argument("--download_dir", type=str, default="01_data_collection/raw_videos",
                       help="Download directory")
    parser.add_argument("--max_videos_per_channel", type=int, default=10,
                       help="Max videos per channel")
    parser.add_argument("--max_videos_per_class", type=int, default=5,
                       help="Max videos per class")
    parser.add_argument("--m3u8_duration", type=int, default=30,
                       help="M3U8 stream duration")
    parser.add_argument("--output_json", type=str, default="automated_download_results.json",
                       help="Output JSON file for results")
    
    args = parser.parse_args()
    
    # Create downloader
    downloader = AutomatedVideoDownloader(args.download_dir)
    
    # Run automated download
    results = downloader.run_automated_download(
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
