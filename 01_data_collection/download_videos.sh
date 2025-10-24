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

# Main execution
main() {
    echo "🚀 Starting Football Video Data Collection"
    echo "📁 Download directory: $DOWNLOAD_DIR"
    echo "🎯 Max videos per class: $MAX_VIDEOS_PER_CLASS"
    echo "⏱️  Video duration: $VIDEO_DURATION seconds"
    echo "📺 Video quality: $VIDEO_QUALITY"
    echo ""
    
    # Download videos for each class
    for class in "${!CLASS_SEARCHES[@]}"; do
        download_class_videos "$class" "${CLASS_SEARCHES[$class]}"
        echo ""
    done
    
    # Download from channels
    download_from_channels
    echo ""
    
    # Download from playlists
    download_from_playlists
    echo ""
    
    # Summary
    echo "📊 Download Summary:"
    for class in "${!CLASS_SEARCHES[@]}"; do
        local count=$(find "$DOWNLOAD_DIR/$class" -name "*.mp4" -o -name "*.webm" -o -name "*.mkv" | wc -l)
        echo "  $class: $count videos"
    done
    
    local total=$(find "$DOWNLOAD_DIR" -name "*.mp4" -o -name "*.webm" -o -name "*.mkv" | wc -l)
    echo "  Total: $total videos"
    echo ""
    echo "✅ Data collection completed!"
}

# Run main function
main "$@"