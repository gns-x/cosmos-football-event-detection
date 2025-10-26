#!/bin/bash

# Football Video Data Collection Script
# Downloads football videos for 8 classes using yt-dlp

set -e

# Configuration
DOWNLOAD_DIR="../01_data_collection/raw_videos"
MAX_VIDEOS_PER_CLASS=10
VIDEO_DURATION="10-30"  # 10-30 seconds clips
VIDEO_QUALITY="best[height<=720]"  # Max 720p for efficiency

# Create download directory
mkdir -p "$DOWNLOAD_DIR"

# Define search terms for each class
declare -A CLASS_SEARCHES=(
    ["penalty_shot"]="football penalty kick goal"
    ["goal"]="football goal scoring"
    ["red_card"]="football red card referee"
    ["yellow_card"]="football yellow card referee"
    ["corner_kick"]="football corner kick"
    ["free_kick"]="football free kick"
    ["throw_in"]="football throw in"
    ["offside"]="football offside VAR"
)

# Function to download videos for a class
download_class_videos() {
    local class_name="$1"
    local search_terms="$2"
    local class_dir="$DOWNLOAD_DIR/$class_name"
    
    echo "🎯 Downloading videos for class: $class_name"
    echo "🔍 Search terms: $search_terms"
    
    # Create class directory
    mkdir -p "$class_dir"
    
    # Search and download videos
    yt-dlp \
        --max-downloads "$MAX_VIDEOS_PER_CLASS" \
        --format "$VIDEO_QUALITY" \
        --match-filter "duration > 10 & duration < 30" \
        --output "$class_dir/%(title)s.%(ext)s" \
        --write-info-json \
        --write-thumbnail \
        --extract-flat \
        "ytsearch10:$search_terms"
    
    echo "✅ Downloaded videos for $class_name"
}

# Function to download from M3U8 streams
download_m3u8_streams() {
    echo "📡 Downloading from M3U8 streams..."
    
    # M3U8 stream sources for football videos
    local m3u8_streams=(
        "https://h2ddsukdqkoh6b.data.mediastore.us-east-1.amazonaws.com/ucl-walionpapl-aug13-eb643a9-archive/master/ucl-walionpapl-aug13-eb643a9-archive.m3u8"
        # Add more M3U8 streams here
    )
    
    for stream_url in "${m3u8_streams[@]}"; do
        echo "📺 Downloading from M3U8 stream: $stream_url"
        
        # Create M3U8 directory
        mkdir -p "$DOWNLOAD_DIR/m3u8_streams"
        
        # Download using ffmpeg
        local output_file="$DOWNLOAD_DIR/m3u8_streams/stream_$(date +%s).mp4"
        
        ffmpeg -i "$stream_url" \
            -t 30 \
            -c copy \
            -bsf:a aac_adtstoasc \
            -y \
            "$output_file" || echo "⚠️  M3U8 download failed for: $stream_url"
        
        if [ -f "$output_file" ]; then
            echo "✅ M3U8 stream downloaded: $output_file"
        fi
    done
}

# Function to download from specific channels (more reliable)
download_from_channels() {
    echo "📺 Downloading from specific football channels..."
    
    # Popular football channels
    local channels=(
        "https://www.youtube.com/@FIFA"
        "https://www.youtube.com/@UEFA"
        "https://www.youtube.com/@PremierLeague"
        "https://www.youtube.com/@LaLiga"
        "https://www.youtube.com/@SerieA"
    )
    
    for channel in "${channels[@]}"; do
        echo "📡 Downloading from channel: $channel"
        yt-dlp \
            --max-downloads 5 \
            --format "$VIDEO_QUALITY" \
            --match-filter "duration > 10 & duration < 30" \
            --output "$DOWNLOAD_DIR/channels/%(uploader)s_%(title)s.%(ext)s" \
            --write-info-json \
            "$channel"
    done
}

# Function to download from playlists
download_from_playlists() {
    echo "📋 Downloading from football playlists..."
    
    # Football highlight playlists
    local playlists=(
        "https://www.youtube.com/playlist?list=PLCGIzmTE4d0hjpV5jCjJdJjJdJjJdJjJdJ"  # Example playlist
    )
    
    for playlist in "${playlists[@]}"; do
        echo "📝 Downloading from playlist: $playlist"
        yt-dlp \
            --max-downloads 10 \
            --format "$VIDEO_QUALITY" \
            --match-filter "duration > 10 & duration < 30" \
            --output "$DOWNLOAD_DIR/playlists/%(playlist_title)s_%(title)s.%(ext)s" \
            --write-info-json \
            "$playlist"
    done
}

# Function to run enhanced Python downloader
run_enhanced_downloader() {
    echo "🚀 Running Enhanced Video Downloader"
    echo "📁 Download directory: $DOWNLOAD_DIR"
    echo "🎯 Max videos per channel: 10"
    echo "🎯 Max videos per class: 5"
    echo "⏱️  M3U8 duration: 30 seconds"
    echo ""
    
    # Run the enhanced Python downloader
    python3 ../scripts/enhanced_video_downloader.py \
        --download_dir "$DOWNLOAD_DIR" \
        --max_videos_per_channel 10 \
        --max_videos_per_class 5 \
        --m3u8_duration 30 \
        --output_json "$DOWNLOAD_DIR/download_results.json"
    
    echo ""
    echo "✅ Enhanced download completed!"
}

# Main execution
main() {
    echo "🚀 Starting Football Video Data Collection"
    echo "📁 Download directory: $DOWNLOAD_DIR"
    echo "🎯 Max videos per class: $MAX_VIDEOS_PER_CLASS"
    echo "⏱️  Video duration: $VIDEO_DURATION seconds"
    echo "📺 Video quality: $VIDEO_QUALITY"
    echo ""
    
    # Check if enhanced downloader is available
    if [ -f "../scripts/enhanced_video_downloader.py" ]; then
        echo "🔧 Using Enhanced Video Downloader (Python)"
        run_enhanced_downloader
    else
        echo "🔧 Using Basic Download Script (Bash)"
        
        # Download videos for each class
        for class in "${!CLASS_SEARCHES[@]}"; do
            download_class_videos "$class" "${CLASS_SEARCHES[$class]}"
            echo ""
        done
        
        # Download from M3U8 streams
        download_m3u8_streams
        echo ""
        
        # Download from channels
        download_from_channels
        echo ""
        
        # Download from playlists
        download_from_playlists
        echo ""
    fi
    
    # Summary
    echo "📊 Download Summary:"
    for class in "${!CLASS_SEARCHES[@]}"; do
        local count=$(find "$DOWNLOAD_DIR/$class" -name "*.mp4" -o -name "*.webm" -o -name "*.mkv" | wc -l)
        echo "  $class: $count videos"
    done
    
    # Check for other directories
    if [ -d "$DOWNLOAD_DIR/channels" ]; then
        local channel_count=$(find "$DOWNLOAD_DIR/channels" -name "*.mp4" -o -name "*.webm" -o -name "*.mkv" | wc -l)
        echo "  channels: $channel_count videos"
    fi
    
    if [ -d "$DOWNLOAD_DIR/m3u8_streams" ]; then
        local m3u8_count=$(find "$DOWNLOAD_DIR/m3u8_streams" -name "*.mp4" -o -name "*.webm" -o -name "*.mkv" | wc -l)
        echo "  m3u8_streams: $m3u8_count videos"
    fi
    
    local total=$(find "$DOWNLOAD_DIR" -name "*.mp4" -o -name "*.webm" -o -name "*.mkv" | wc -l)
    echo "  Total: $total videos"
    echo ""
    echo "✅ Data collection completed!"
}

# Run main function
main "$@"