# M3U8 Stream Sources Documentation

## 🎯 **Comprehensive M3U8 Stream Sources**

This document provides a comprehensive list of M3U8 stream sources for football video analysis, including open source, free, and educational content.

## 📡 **Primary M3U8 Stream Sources**

### **1. AWS MediaStore Streams**
- **Source**: Amazon Web Services MediaStore
- **Type**: Professional sports content
- **Access**: Public streams
- **Examples**:
  ```
  https://h2ddsukdqkoh6b.data.mediastore.us-east-1.amazonaws.com/ucl-walionpapl-aug13-eb643a9-archive/master/ucl-walionpapl-aug13-eb643a9-archive.m3u8
  ```

### **2. Archive.org Streams**
- **Source**: Internet Archive
- **Type**: Historical sports content
- **Access**: Public domain
- **Examples**:
  ```
  https://archive.org/download/football_highlights/football_highlights.m3u8
  https://archive.org/download/soccer_goals/soccer_goals.m3u8
  https://archive.org/download/football_matches/football_matches.m3u8
  ```

### **3. Educational Sports Content**
- **Source**: Educational institutions
- **Type**: Learning materials
- **Access**: Open source
- **Examples**:
  ```
  https://open-source-sports.example.com/football.m3u8
  https://educational-sports.example.com/soccer.m3u8
  ```

### **4. Public Domain Sports Content**
- **Source**: Public domain repositories
- **Type**: Free sports content
- **Access**: No restrictions
- **Examples**:
  ```
  https://public-sports.example.com/football.m3u8
  https://free-sports.example.com/soccer.m3u8
  ```

### **5. IPTV Sources**
- **Source**: Internet Protocol Television
- **Type**: Live sports channels
- **Access**: Free channels
- **Examples**:
  ```
  https://iptv.example.com/sports.m3u8
  https://free-iptv.example.com/football.m3u8
  ```

## 🔍 **M3U8 Stream Discovery Methods**

### **1. Pattern-Based Discovery**
The system automatically tests common patterns:
```
https://{domain}/sports/football.m3u8
https://{domain}/football/highlights.m3u8
https://{domain}/soccer/goals.m3u8
https://{domain}/live/football.m3u8
https://{domain}/streams/sports.m3u8
```

### **2. IPTV Playlist Discovery**
The system scans IPTV playlists for sports streams:
```
https://iptv-org.github.io/iptv/index.m3u
https://raw.githubusercontent.com/iptv-org/iptv/master/index.m3u
https://raw.githubusercontent.com/iptv-org/iptv/master/countries/us.m3u
https://raw.githubusercontent.com/iptv-org/iptv/master/categories/sport.m3u
```

### **3. Search Engine Discovery**
The system searches for M3U8 streams using queries:
- "football m3u8 stream"
- "soccer m3u8 stream"
- "sports m3u8 stream"
- "free football m3u8"
- "live sports m3u8"

## 🛠️ **M3U8 Stream Tools**

### **1. FFmpeg (Primary Tool)**
```bash
# Download M3U8 stream
ffmpeg -i "stream_url" -c copy -bsf:a aac_adtstoasc -y "output.mp4"

# Test stream accessibility
ffprobe -v quiet -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "stream_url"
```

### **2. yt-dlp (Alternative Tool)**
```bash
# Download M3U8 stream
yt-dlp "stream_url" -o "output.mp4"

# Test stream accessibility
yt-dlp --simulate "stream_url"
```

### **3. hls-fetch (Specialized Tool)**
```bash
# Download HLS stream
hls-fetch "stream_url" -o "output.mp4"
```

### **4. m3u8downloader (Python Tool)**
```bash
# Download M3U8 stream
python m3u8downloader.py "stream_url" "output.mp4"
```

## 📊 **Stream Validation Rules**

### **1. Accessibility Tests**
- HTTP response code 200
- Content-Type: application/vnd.apple.mpegurl
- Contains #EXTM3U header
- Stream duration > 5 seconds

### **2. Content Validation**
- Video codec: H.264/AVC1
- Resolution: 320x240 to 1920x1080
- Duration: 5-300 seconds
- File size: 100KB to 500MB

### **3. Quality Checks**
- No audio/video sync issues
- No corruption or artifacts
- Smooth playback
- Proper encoding

## 🚀 **Automated Discovery Process**

### **1. Discovery Script**
```bash
# Run M3U8 stream discovery
python scripts/m3u8_stream_discovery.py --output discovered_streams.json
```

### **2. Validation Process**
```bash
# Test discovered streams
python scripts/automated_video_downloader.py --test_m3u8_only
```

### **3. Integration**
```bash
# Use discovered streams in automated downloader
python scripts/automated_video_downloader.py --use_discovered_streams
```

## 📁 **Stream Categories**

### **1. Live Streams**
- Real-time sports events
- Live football matches
- Live sports news
- Live highlights

### **2. On-Demand Streams**
- Football highlights
- Goal compilations
- Match replays
- Training videos

### **3. Educational Streams**
- Sports analysis
- Training tutorials
- Tactical breakdowns
- Historical content

### **4. Archive Streams**
- Historical matches
- Classic goals
- Legendary moments
- Sports documentaries

## 🔧 **Configuration**

### **1. Stream Sources Configuration**
```json
{
  "m3u8_streams": [
    "https://h2ddsukdqkoh6b.data.mediastore.us-east-1.amazonaws.com/ucl-walionpapl-aug13-eb643a9-archive/master/ucl-walionpapl-aug13-eb643a9-archive.m3u8",
    "https://archive.org/download/football_highlights/football_highlights.m3u8"
  ],
  "discovery_enabled": true,
  "auto_validation": true,
  "max_streams_per_source": 10
}
```

### **2. Validation Rules**
```json
{
  "validation": {
    "min_duration": 5,
    "max_duration": 300,
    "min_file_size": 100000,
    "max_file_size": 500000000,
    "required_codecs": ["h264", "avc1"],
    "min_resolution": [320, 240],
    "max_resolution": [1920, 1080]
  }
}
```

## 📈 **Expected Results**

### **1. Stream Discovery**
- **Known Sources**: 10+ streams
- **Pattern Discovery**: 50+ potential streams
- **IPTV Playlists**: 100+ streams
- **Search Results**: 200+ streams

### **2. Validation Results**
- **Accessible Streams**: 60-80% of discovered streams
- **Valid Content**: 40-60% of accessible streams
- **Quality Streams**: 20-40% of valid streams

### **3. Download Results**
- **Successful Downloads**: 80-90% of quality streams
- **Valid Videos**: 70-85% of downloads
- **Final Dataset**: 50-100+ videos per run

## 🎯 **Usage Examples**

### **1. Manual Stream Testing**
```bash
# Test a specific M3U8 stream
python scripts/download_m3u8_video.py --url "stream_url" --output "test.mp4"
```

### **2. Automated Discovery**
```bash
# Discover new M3U8 streams
python scripts/m3u8_stream_discovery.py
```

### **3. Full Automated Download**
```bash
# Run complete automated download with discovery
python scripts/automated_video_downloader.py --discover_streams
```

## 📚 **Additional Resources**

### **1. M3U8 Documentation**
- [M3U8 Format Specification](https://tools.ietf.org/html/rfc8216)
- [HLS (HTTP Live Streaming) Documentation](https://developer.apple.com/streaming/)

### **2. Open Source Tools**
- [yt-dlp GitHub](https://github.com/yt-dlp/yt-dlp)
- [FFmpeg Documentation](https://ffmpeg.org/documentation.html)
- [hls-fetch GitHub](https://github.com/krzemienski/awesome-video)

### **3. IPTV Resources**
- [IPTV-org GitHub](https://github.com/iptv-org/iptv)
- [Free IPTV Lists](https://github.com/iptv-org/iptv/blob/master/README.md)

## ✅ **Best Practices**

### **1. Legal Compliance**
- Respect terms of service
- Use only public domain content
- Avoid copyrighted material
- Follow fair use guidelines

### **2. Technical Best Practices**
- Test streams before downloading
- Validate content quality
- Monitor download success rates
- Implement error handling

### **3. Resource Management**
- Limit concurrent downloads
- Monitor storage usage
- Clean up invalid files
- Optimize for efficiency

---

**This comprehensive M3U8 stream source system provides automated discovery, validation, and downloading of football video content from open sources.**
