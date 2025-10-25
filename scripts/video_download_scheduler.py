#!/usr/bin/env python3
"""
Video Download Scheduler
Automated scheduling and rule-based video downloading
"""

import os
import sys
import json
import time
import schedule
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List
import argparse
import subprocess


class VideoDownloadScheduler:
    """Automated video download scheduler with rules."""
    
    def __init__(self, config_file: str = "download_rules.json"):
        self.config_file = Path(config_file)
        self.setup_logging()
        self.load_config()
        
    def setup_logging(self):
        """Setup logging for the scheduler."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"scheduler_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def load_config(self):
        """Load download rules configuration."""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
        else:
            # Default configuration
            self.config = {
                "download_rules": {
                    "enabled": True,
                    "schedule": "daily",
                    "time": "02:00",
                    "max_videos_per_channel": 10,
                    "max_videos_per_class": 5,
                    "m3u8_duration": 30,
                    "auto_cleanup": True,
                    "validation_enabled": True
                },
                "sources": {
                    "channels": [
                        "FIFA", "UEFA", "PremierLeague", "LaLiga", "SerieA",
                        "Bundesliga", "Ligue1", "ChampionsLeague", "EuropaLeague", "WorldCup"
                    ],
                    "search_terms": [
                        "penalty_shot", "goal", "red_card", "yellow_card",
                        "corner_kick", "free_kick", "throw_in", "offside"
                    ],
                    "m3u8_streams": [
                        "https://h2ddsukdqkoh6b.data.mediastore.us-east-1.amazonaws.com/ucl-walionpapl-aug13-eb643a9-archive/master/ucl-walionpapl-aug13-eb643a9-archive.m3u8"
                    ]
                },
                "validation": {
                    "min_duration": 5,
                    "max_duration": 300,
                    "min_file_size": 100000,
                    "max_file_size": 500000000,
                    "required_codecs": ["h264", "avc1"],
                    "min_resolution": [320, 240],
                    "max_resolution": [1920, 1080]
                },
                "cleanup": {
                    "remove_invalid": True,
                    "max_storage_gb": 50,
                    "cleanup_old_videos_days": 30
                }
            }
            self.save_config()
    
    def save_config(self):
        """Save configuration to file."""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def run_automated_download(self):
        """Run the automated download process."""
        try:
            self.logger.info("🚀 Starting automated video download...")
            
            # Run the automated downloader
            cmd = [
                "python", "scripts/automated_video_downloader.py",
                "--download_dir", "01_data_collection/raw_videos",
                "--max_videos_per_channel", str(self.config["download_rules"]["max_videos_per_channel"]),
                "--max_videos_per_class", str(self.config["download_rules"]["max_videos_per_class"]),
                "--m3u8_duration", str(self.config["download_rules"]["m3u8_duration"]),
                "--output_json", f"download_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            ]
            
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            self.logger.info("✅ Automated download completed successfully")
            
            # Run cleanup if enabled
            if self.config["download_rules"]["auto_cleanup"]:
                self.run_cleanup()
            
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"❌ Automated download failed: {e}")
            self.logger.error(f"Error output: {e.stderr}")
            return False
        except Exception as e:
            self.logger.error(f"❌ Download error: {e}")
            return False
    
    def run_cleanup(self):
        """Run cleanup operations."""
        try:
            self.logger.info("🧹 Running cleanup operations...")
            
            # Remove invalid videos
            if self.config["cleanup"]["remove_invalid"]:
                self.remove_invalid_videos()
            
            # Check storage usage
            self.check_storage_usage()
            
            # Clean old videos
            self.clean_old_videos()
            
            self.logger.info("✅ Cleanup completed")
            
        except Exception as e:
            self.logger.error(f"❌ Cleanup error: {e}")
    
    def remove_invalid_videos(self):
        """Remove invalid videos from the download directory."""
        download_dir = Path("01_data_collection/raw_videos")
        removed_count = 0
        
        for video_file in download_dir.rglob("*.mp4"):
            if video_file.is_file():
                # Check if video is valid using ffprobe
                try:
                    cmd = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", 
                           "-of", "default=noprint_wrappers=1:nokey=1", str(video_file)]
                    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                    duration = float(result.stdout.strip())
                    
                    # Remove if duration is too short or too long
                    if duration < self.config["validation"]["min_duration"] or duration > self.config["validation"]["max_duration"]:
                        video_file.unlink()
                        removed_count += 1
                        self.logger.info(f"🗑️  Removed invalid video: {video_file.name}")
                        
                except Exception as e:
                    # Remove if ffprobe fails
                    video_file.unlink()
                    removed_count += 1
                    self.logger.info(f"🗑️  Removed corrupted video: {video_file.name}")
        
        self.logger.info(f"✅ Removed {removed_count} invalid videos")
    
    def check_storage_usage(self):
        """Check storage usage and clean if necessary."""
        download_dir = Path("01_data_collection/raw_videos")
        max_storage_gb = self.config["cleanup"]["max_storage_gb"]
        
        # Calculate total size
        total_size = 0
        for video_file in download_dir.rglob("*.mp4"):
            if video_file.is_file():
                total_size += video_file.stat().st_size
        
        total_size_gb = total_size / (1024**3)
        
        if total_size_gb > max_storage_gb:
            self.logger.warning(f"⚠️  Storage usage: {total_size_gb:.2f}GB > {max_storage_gb}GB")
            self.logger.info("🧹 Cleaning old videos to free space...")
            self.clean_old_videos()
        else:
            self.logger.info(f"✅ Storage usage: {total_size_gb:.2f}GB < {max_storage_gb}GB")
    
    def clean_old_videos(self):
        """Clean old videos based on age."""
        download_dir = Path("01_data_collection/raw_videos")
        cleanup_days = self.config["cleanup"]["cleanup_old_videos_days"]
        cutoff_date = datetime.now() - timedelta(days=cleanup_days)
        
        removed_count = 0
        for video_file in download_dir.rglob("*.mp4"):
            if video_file.is_file():
                file_time = datetime.fromtimestamp(video_file.stat().st_mtime)
                if file_time < cutoff_date:
                    video_file.unlink()
                    removed_count += 1
                    self.logger.info(f"🗑️  Removed old video: {video_file.name}")
        
        self.logger.info(f"✅ Removed {removed_count} old videos")
    
    def setup_schedule(self):
        """Setup the download schedule."""
        schedule_type = self.config["download_rules"]["schedule"]
        schedule_time = self.config["download_rules"]["time"]
        
        if schedule_type == "daily":
            schedule.every().day.at(schedule_time).do(self.run_automated_download)
            self.logger.info(f"📅 Scheduled daily download at {schedule_time}")
        elif schedule_type == "weekly":
            schedule.every().monday.at(schedule_time).do(self.run_automated_download)
            self.logger.info(f"📅 Scheduled weekly download on Monday at {schedule_time}")
        elif schedule_type == "hourly":
            schedule.every().hour.do(self.run_automated_download)
            self.logger.info("📅 Scheduled hourly download")
        else:
            self.logger.error(f"❌ Unknown schedule type: {schedule_type}")
            return False
        
        return True
    
    def run_scheduler(self):
        """Run the scheduler."""
        if not self.config["download_rules"]["enabled"]:
            self.logger.info("⏸️  Scheduler disabled in configuration")
            return
        
        self.logger.info("🚀 Starting Video Download Scheduler")
        self.logger.info("=" * 50)
        
        if not self.setup_schedule():
            return
        
        # Run initial download
        self.logger.info("🔄 Running initial download...")
        self.run_automated_download()
        
        # Keep scheduler running
        self.logger.info("⏰ Scheduler running... Press Ctrl+C to stop")
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            self.logger.info("⏹️  Scheduler stopped by user")
    
    def run_once(self):
        """Run download once without scheduling."""
        self.logger.info("🚀 Running one-time download...")
        return self.run_automated_download()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Video Download Scheduler")
    parser.add_argument("--config", type=str, default="download_rules.json",
                       help="Configuration file")
    parser.add_argument("--once", action="store_true",
                       help="Run once without scheduling")
    parser.add_argument("--setup", action="store_true",
                       help="Setup initial configuration")
    
    args = parser.parse_args()
    
    # Create scheduler
    scheduler = VideoDownloadScheduler(args.config)
    
    if args.setup:
        print("🔧 Setting up initial configuration...")
        scheduler.save_config()
        print(f"✅ Configuration saved to: {args.config}")
        return 0
    
    if args.once:
        # Run once
        success = scheduler.run_once()
        return 0 if success else 1
    else:
        # Run scheduler
        scheduler.run_scheduler()
        return 0


if __name__ == "__main__":
    exit(main())
