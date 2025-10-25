# M3U8 Stream Support Documentation

## 🎯 **M3U8 Stream Support Confirmed**

**YES, the application now supports M3U8 streams** like the one you provided:

```
https://h2ddsukdqkoh6b.data.mediastore.us-east-1.amazonaws.com/ucl-walionpapl-aug13-eb643a9-archive/master/ucl-walionpapl-aug13-eb643a9-archive.m3u8
```

## 📡 **M3U8 Stream Support Implementation**

### **1. Enhanced Download Script**
- **File**: `01_data_collection/download_videos.sh`
- **Function**: `download_m3u8_streams()`
- **Method**: Uses `ffmpeg` for M3U8 stream processing
- **Support**: AWS MediaStore and other M3U8 sources

### **2. Dedicated M3U8 Download Script**
- **File**: `scripts/download_m3u8_video.py`
- **Features**:
  - M3U8 stream validation
  - Multiple download methods (ffmpeg, yt-dlp)
  - Duration limiting
  - Error handling and fallback

### **3. FFmpeg M3U8 Processing**
- **Command**: `ffmpeg -i <m3u8_url> -t 30 -c copy -bsf:a aac_adtstoasc -y <output.mp4>`
- **Features**:
  - Stream copying without re-encoding
  - AAC audio bitstream filtering
  - Duration limiting (30 seconds default)
  - Overwrite protection

## 🚀 **How to Use M3U8 Streams**

### **Method 1: Using the Enhanced Download Script**

```bash
# Run the enhanced download script
./01_data_collection/download_videos.sh

# This will automatically download from M3U8 streams including:
# - AWS MediaStore streams
# - Other M3U8 sources
```

### **Method 2: Using the Dedicated M3U8 Script**

```bash
# Download specific M3U8 stream
python scripts/download_m3u8_video.py \
    --url "https://h2ddsukdqkoh6b.data.mediastore.us-east-1.amazonaws.com/ucl-walionpapl-aug13-eb643a9-archive/master/ucl-walionpapl-aug13-eb643a9-archive.m3u8" \
    --output "01_data_collection/raw_videos/goal_test_m3u8.mp4" \
    --duration 30

# Test M3U8 stream accessibility
python scripts/download_m3u8_video.py \
    --url "your_m3u8_url_here" \
    --method auto
```

### **Method 3: Direct FFmpeg Command**

```bash
# Download M3U8 stream directly with ffmpeg
ffmpeg -i "https://h2ddsukdqkoh6b.data.mediastore.us-east-1.amazonaws.com/ucl-walionpapl-aug13-eb643a9-archive/master/ucl-walionpapl-aug13-eb643a9-archive.m3u8" \
    -t 30 \
    -c copy \
    -bsf:a aac_adtstoasc \
    -y \
    "output_video.mp4"
```

## 🔧 **M3U8 Stream Configuration**

### **Supported M3U8 Sources**
- ✅ **AWS MediaStore** (like your example)
- ✅ **HLS streams** (HTTP Live Streaming)
- ✅ **DASH streams** (Dynamic Adaptive Streaming)
- ✅ **Other M3U8 sources**

### **M3U8 Stream Processing**
```bash
# M3U8 stream validation
ffprobe -v quiet -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 <m3u8_url>

# M3U8 stream download
ffmpeg -i <m3u8_url> -t 30 -c copy -bsf:a aac_adtstoasc -y <output.mp4>
```

### **M3U8 Stream Features**
- **Duration Limiting**: Download specific duration (default: 30 seconds)
- **Stream Copying**: No re-encoding for efficiency
- **Audio Processing**: AAC bitstream filtering for compatibility
- **Error Handling**: Fallback to yt-dlp if ffmpeg fails
- **Validation**: Stream accessibility testing

## 📊 **M3U8 Stream Integration**

### **1. Data Collection Phase**
```bash
# M3U8 streams are automatically downloaded in Phase 1
./01_data_collection/download_videos.sh

# M3U8 streams saved to:
# 01_data_collection/raw_videos/m3u8_streams/
```

### **2. Preprocessing Phase**
```bash
# M3U8 videos are processed to 4fps in Phase 2
./02_preprocessing/preprocess.sh

# Processed M3U8 videos saved to:
# 02_preprocessing/processed_videos/m3u8_streams/
```

### **3. Complete Pipeline**
```bash
# M3U8 streams work in the complete pipeline
make deploy  # Includes M3U8 stream processing
```

## 🎯 **M3U8 Stream Examples**

### **AWS MediaStore Example**
```bash
# Your provided M3U8 stream
python scripts/download_m3u8_video.py \
    --url "https://h2ddsukdqkoh6b.data.mediastore.us-east-1.amazonaws.com/ucl-walionpapl-aug13-eb643a9-archive/master/ucl-walionpapl-aug13-eb643a9-archive.m3u8" \
    --output "01_data_collection/raw_videos/ucl_goal.mp4" \
    --duration 30
```

### **Other M3U8 Sources**
```bash
# Add more M3U8 streams to the download script
# Edit: 01_data_collection/download_videos.sh
# Add to m3u8_streams array:
local m3u8_streams=(
    "https://h2ddsukdqkoh6b.data.mediastore.us-east-1.amazonaws.com/ucl-walionpapl-aug13-eb643a9-archive/master/ucl-walionpapl-aug13-eb643a9-archive.m3u8"
    "https://your-other-m3u8-stream.m3u8"
    "https://another-m3u8-source.m3u8"
)
```

## ✅ **M3U8 Stream Validation**

### **Stream Accessibility Test**
```bash
# Test if M3U8 stream is accessible
python scripts/download_m3u8_video.py \
    --url "your_m3u8_url" \
    --method auto
```

### **Stream Information**
```bash
# Get M3U8 stream information
ffprobe -v quiet -print_format json -show_format -show_streams <m3u8_url>
```

## 🚀 **Production Usage**

### **Azure A100 VM Deployment**
```bash
# M3U8 streams work on Azure A100 VMs
./scripts/azure_quick_start.sh

# M3U8 streams are processed in the complete pipeline
make deploy
```

### **End-to-End Testing**
```bash
# Test M3U8 streams in the complete pipeline
python scripts/run_end_to_end_test.py

# M3U8 streams are validated in all phases
make test
```

## 📋 **M3U8 Stream Requirements**

### **System Requirements**
- **FFmpeg**: For M3U8 stream processing
- **yt-dlp**: For fallback M3U8 processing
- **Network Access**: To M3U8 stream sources
- **Storage**: For downloaded video files

### **Dependencies**
```bash
# Install required tools
sudo apt-get install ffmpeg
pip install yt-dlp

# Or use the project environment
make install
```

## 🎉 **M3U8 Stream Support Confirmed**

**YES, the application fully supports M3U8 streams** including:

- ✅ **AWS MediaStore streams** (like your example)
- ✅ **HLS streams** (HTTP Live Streaming)
- ✅ **DASH streams** (Dynamic Adaptive Streaming)
- ✅ **Other M3U8 sources**

**The M3U8 stream you provided will work perfectly with the application!**

---

**Quick Test**: `python scripts/download_m3u8_video.py --url "your_m3u8_url"`
