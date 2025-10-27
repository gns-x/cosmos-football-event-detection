#!/usr/bin/env python3
"""
Sample Football Video Data Collection - DEPRECATED
NOTE: This file creates placeholder videos for testing. For real video collection,
use: scripts/download_specific_football_events.py

This file is kept for compatibility but should not be used in production.
"""

import os
import subprocess
import json
from pathlib import Path
from typing import Dict, List
import argparse

class SampleDataCollector:
    """
    DEPRECATED: This class creates placeholder videos.
    Use scripts/download_specific_football_events.py for real video downloads.
    """
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.raw_videos_dir = self.project_root / "01_data_collection" / "raw_videos"
        self.processed_videos_dir = self.project_root / "02_preprocessing" / "processed_videos"
        
        print("⚠️  WARNING: This script creates PLACEHOLDER videos!")
        print("⚠️  For real football videos, use: scripts/download_specific_football_events.py")
        
        # Create directories
        self.raw_videos_dir.mkdir(parents=True, exist_ok=True)
        self.processed_videos_dir.mkdir(parents=True, exist_ok=True)
    
    def collect_sample_data(self):
        """
        DEPRECATED: Creates placeholder videos for testing only.
        For production, use the real video download script.
        """
        print("=" * 70)
        print("⚠️  DEPRECATED: This creates PLACEHOLDER videos!")
        print("⚠️  For real video collection, use:")
        print("   make download-videos")
        print("   OR")
        print("   python scripts/download_specific_football_events.py")
        print("=" * 70)
        return

def main():
    parser = argparse.ArgumentParser(
        description="DEPRECATED: Collect Sample Football Video Data",
        epilog="Note: This script creates placeholder videos. Use scripts/download_specific_football_events.py for real downloads."
    )
    parser.add_argument("--project-root", default=".",
                       help="Project root directory")
    
    args = parser.parse_args()
    
    collector = SampleDataCollector(args.project_root)
    collector.collect_sample_data()

if __name__ == "__main__":
    main()
