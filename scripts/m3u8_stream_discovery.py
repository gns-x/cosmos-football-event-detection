#!/usr/bin/env python3
"""
M3U8 Stream Discovery and Validation
Discovers and validates M3U8 streams from various sources
"""

import os
import subprocess
import sys
import json
import time
import requests
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import argparse
from datetime import datetime
import logging


class M3U8StreamDiscovery:
    """Discovers and validates M3U8 streams from various sources."""
    
    def __init__(self, output_file: str = "discovered_m3u8_streams.json"):
        self.output_file = Path(output_file)
        self.setup_logging()
        
        # Known M3U8 stream sources
        self.known_sources = {
            "aws_mediastore": [
                "https://h2ddsukdqkoh6b.data.mediastore.us-east-1.amazonaws.com/ucl-walionpapl-aug13-eb643a9-archive/master/ucl-walionpapl-aug13-eb643a9-archive.m3u8"
            ],
            "archive_org": [
                "https://archive.org/download/football_highlights/football_highlights.m3u8",
                "https://archive.org/download/soccer_goals/soccer_goals.m3u8",
                "https://archive.org/download/football_matches/football_matches.m3u8"
            ],
            "educational": [
                "https://open-source-sports.example.com/football.m3u8",
                "https://educational-sports.example.com/soccer.m3u8"
            ],
            "public_domain": [
                "https://public-sports.example.com/football.m3u8",
                "https://free-sports.example.com/soccer.m3u8"
            ],
            "iptv": [
                "https://iptv.example.com/sports.m3u8",
                "https://free-iptv.example.com/football.m3u8"
            ]
        }
        
        # Discovery patterns
        self.discovery_patterns = [
            "https://{domain}/sports/football.m3u8",
            "https://{domain}/football/highlights.m3u8",
            "https://{domain}/soccer/goals.m3u8",
            "https://{domain}/live/football.m3u8",
            "https://{domain}/streams/sports.m3u8"
        ]
        
        # Common domains to test
        self.test_domains = [
            "archive.org",
            "stream.example.com",
            "sports.example.com",
            "live.example.com",
            "iptv.example.com",
            "free-sports.example.com",
            "public-sports.example.com",
            "open-source-sports.example.com"
        ]
        
        self.discovered_streams = []
        self.valid_streams = []
        self.invalid_streams = []
    
    def setup_logging(self):
        """Setup logging for the discovery process."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"m3u8_discovery_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def test_m3u8_stream(self, stream_url: str) -> Tuple[bool, Dict[str, any]]:
        """Test if an M3U8 stream is accessible and valid."""
        try:
            self.logger.info(f"🔍 Testing M3U8 stream: {stream_url}")
            
            # Test with ffprobe first
            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                stream_url
            ]
            
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=30)
            duration = float(result.stdout.strip())
            
            # Test with requests to check HTTP accessibility
            response = requests.get(stream_url, timeout=10)
            response.raise_for_status()
            
            # Check if it's a valid M3U8 file
            if "#EXTM3U" in response.text:
                stream_info = {
                    "url": stream_url,
                    "duration": duration,
                    "content_type": response.headers.get("content-type", "unknown"),
                    "content_length": response.headers.get("content-length", "unknown"),
                    "accessible": True,
                    "valid_m3u8": True,
                    "tested_at": datetime.now().isoformat()
                }
                
                self.logger.info(f"✅ Valid M3U8 stream: {stream_url}")
                return True, stream_info
            else:
                self.logger.warning(f"⚠️  Not a valid M3U8 file: {stream_url}")
                return False, {"url": stream_url, "error": "Not a valid M3U8 file"}
                
        except subprocess.CalledProcessError as e:
            self.logger.error(f"❌ FFprobe failed for {stream_url}: {e}")
            return False, {"url": stream_url, "error": f"FFprobe failed: {e}"}
        except requests.RequestException as e:
            self.logger.error(f"❌ HTTP request failed for {stream_url}: {e}")
            return False, {"url": stream_url, "error": f"HTTP request failed: {e}"}
        except Exception as e:
            self.logger.error(f"❌ Error testing {stream_url}: {e}")
            return False, {"url": stream_url, "error": f"Test error: {e}"}
    
    def discover_from_known_sources(self) -> List[Dict[str, any]]:
        """Discover M3U8 streams from known sources."""
        self.logger.info("🔍 Discovering M3U8 streams from known sources...")
        
        discovered = []
        
        for source_type, urls in self.known_sources.items():
            self.logger.info(f"📡 Testing {source_type} sources...")
            
            for url in urls:
                is_valid, info = self.test_m3u8_stream(url)
                
                if is_valid:
                    info["source_type"] = source_type
                    discovered.append(info)
                    self.valid_streams.append(info)
                else:
                    self.invalid_streams.append(info)
        
        return discovered
    
    def discover_from_patterns(self) -> List[Dict[str, any]]:
        """Discover M3U8 streams using pattern matching."""
        self.logger.info("🔍 Discovering M3U8 streams using patterns...")
        
        discovered = []
        
        for domain in self.test_domains:
            self.logger.info(f"🌐 Testing domain: {domain}")
            
            for pattern in self.discovery_patterns:
                test_url = pattern.format(domain=domain)
                is_valid, info = self.test_m3u8_stream(test_url)
                
                if is_valid:
                    info["source_type"] = "pattern_discovery"
                    info["domain"] = domain
                    discovered.append(info)
                    self.valid_streams.append(info)
                else:
                    self.invalid_streams.append(info)
        
        return discovered
    
    def discover_from_playlists(self) -> List[Dict[str, any]]:
        """Discover M3U8 streams from IPTV playlists."""
        self.logger.info("🔍 Discovering M3U8 streams from IPTV playlists...")
        
        # Common IPTV playlist sources
        iptv_sources = [
            "https://iptv-org.github.io/iptv/index.m3u",
            "https://raw.githubusercontent.com/iptv-org/iptv/master/index.m3u",
            "https://raw.githubusercontent.com/iptv-org/iptv/master/countries/us.m3u",
            "https://raw.githubusercontent.com/iptv-org/iptv/master/categories/sport.m3u"
        ]
        
        discovered = []
        
        for playlist_url in iptv_sources:
            try:
                self.logger.info(f"📺 Testing IPTV playlist: {playlist_url}")
                response = requests.get(playlist_url, timeout=10)
                response.raise_for_status()
                
                # Parse M3U playlist
                lines = response.text.split('\n')
                for i, line in enumerate(lines):
                    if line.startswith('http') and '.m3u8' in line:
                        stream_url = line.strip()
                        is_valid, info = self.test_m3u8_stream(stream_url)
                        
                        if is_valid:
                            info["source_type"] = "iptv_playlist"
                            info["playlist_url"] = playlist_url
                            discovered.append(info)
                            self.valid_streams.append(info)
                        else:
                            self.invalid_streams.append(info)
                            
            except Exception as e:
                self.logger.error(f"❌ Error processing playlist {playlist_url}: {e}")
        
        return discovered
    
    def discover_from_search_engines(self) -> List[Dict[str, any]]:
        """Discover M3U8 streams using search engines."""
        self.logger.info("🔍 Discovering M3U8 streams using search engines...")
        
        # Search queries for M3U8 streams
        search_queries = [
            "football m3u8 stream",
            "soccer m3u8 stream",
            "sports m3u8 stream",
            "free football m3u8",
            "live sports m3u8"
        ]
        
        discovered = []
        
        for query in search_queries:
            self.logger.info(f"🔍 Searching for: {query}")
            
            # This is a placeholder - in a real implementation, you would use
            # a search API or web scraping to find M3U8 streams
            # For now, we'll just log the search query
            self.logger.info(f"📝 Search query: {query}")
        
        return discovered
    
    def save_discovered_streams(self):
        """Save discovered streams to JSON file."""
        results = {
            "discovery_info": {
                "total_discovered": len(self.discovered_streams),
                "valid_streams": len(self.valid_streams),
                "invalid_streams": len(self.invalid_streams),
                "discovery_date": datetime.now().isoformat()
            },
            "valid_streams": self.valid_streams,
            "invalid_streams": self.invalid_streams,
            "discovered_streams": self.discovered_streams
        }
        
        with open(self.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"💾 Discovered streams saved to: {self.output_file}")
    
    def run_discovery(self) -> Dict[str, any]:
        """Run the complete M3U8 stream discovery process."""
        self.logger.info("🚀 Starting M3U8 Stream Discovery")
        self.logger.info("=" * 50)
        
        # Discover from known sources
        known_discovered = self.discover_from_known_sources()
        self.discovered_streams.extend(known_discovered)
        
        # Discover from patterns
        pattern_discovered = self.discover_from_patterns()
        self.discovered_streams.extend(pattern_discovered)
        
        # Discover from playlists
        playlist_discovered = self.discover_from_playlists()
        self.discovered_streams.extend(playlist_discovered)
        
        # Discover from search engines
        search_discovered = self.discover_from_search_engines()
        self.discovered_streams.extend(search_discovered)
        
        # Save results
        self.save_discovered_streams()
        
        # Summary
        self.logger.info("📊 Discovery Summary:")
        self.logger.info(f"  Total discovered: {len(self.discovered_streams)}")
        self.logger.info(f"  Valid streams: {len(self.valid_streams)}")
        self.logger.info(f"  Invalid streams: {len(self.invalid_streams)}")
        
        return {
            "total_discovered": len(self.discovered_streams),
            "valid_streams": len(self.valid_streams),
            "invalid_streams": len(self.invalid_streams),
            "discovered_streams": self.discovered_streams
        }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="M3U8 Stream Discovery")
    parser.add_argument("--output", type=str, default="discovered_m3u8_streams.json",
                       help="Output JSON file for discovered streams")
    parser.add_argument("--test_only", action="store_true",
                       help="Test only known sources")
    
    args = parser.parse_args()
    
    # Create discovery instance
    discovery = M3U8StreamDiscovery(args.output)
    
    # Run discovery
    results = discovery.run_discovery()
    
    print(f"\n📄 Results saved to: {args.output}")
    print(f"✅ Discovery completed: {results['valid_streams']} valid streams found")
    
    return 0


if __name__ == "__main__":
    exit(main())
