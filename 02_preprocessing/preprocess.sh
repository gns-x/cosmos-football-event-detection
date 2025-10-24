#!/bin/bash

# Football Video Preprocessing Script
# Processes raw videos to 4 FPS as required by Cosmos-Reason1-7B

set -e

# Configuration
RAW_VIDEOS_DIR="../01_data_collection/raw_videos"
PROCESSED_VIDEOS_DIR="../02_preprocessing/processed_videos"
TARGET_FPS=4
TARGET_RESOLUTION="720x480"  # Standard resolution for efficiency
VIDEO_CODEC="libx264"
AUDIO_CODEC="aac"

# Create processed videos directory
mkdir -p "$PROCESSED_VIDEOS_DIR"

# Function to process a single video
process_video() {
    local input_file="$1"
    local output_file="$2"
    local class_name="$3"
    
    echo "🎬 Processing: $(basename "$input_file")"
    
    # Extract video info
    local duration=$(ffprobe -v quiet -show_entries format=duration -of csv=p=0 "$input_file")
    local fps=$(ffprobe -v quiet -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 "$input_file")
    
    echo "  📊 Original: ${duration}s, ${fps} FPS"
    
    # Process video to 4 FPS
    ffmpeg -i "$input_file" \
        -vf "fps=$TARGET_FPS,scale=$TARGET_RESOLUTION" \
        -c:v "$VIDEO_CODEC" \
        -c:a "$AUDIO_CODEC" \
        -preset fast \
        -crf 23 \
        -y \
        "$output_file"
    
    # Verify output
    local new_duration=$(ffprobe -v quiet -show_entries format=duration -of csv=p=0 "$output_file")
    local new_fps=$(ffprobe -v quiet -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 "$output_file")
    
    echo "  ✅ Processed: ${new_duration}s, ${new_fps} FPS"
    echo "  💾 Saved to: $output_file"
    echo ""
}

# Function to process all videos in a directory
process_directory() {
    local input_dir="$1"
    local output_dir="$2"
    local class_name="$3"
    
    echo "📁 Processing directory: $input_dir"
    echo "📁 Output directory: $output_dir"
    echo "🏷️  Class: $class_name"
    echo ""
    
    # Create output directory for this class
    mkdir -p "$output_dir/$class_name"
    
    # Find all video files
    local video_files=($(find "$input_dir" -type f \( -name "*.mp4" -o -name "*.webm" -o -name "*.mkv" -o -name "*.avi" -o -name "*.mov" \)))
    
    if [ ${#video_files[@]} -eq 0 ]; then
        echo "⚠️  No video files found in $input_dir"
        return
    fi
    
    echo "📹 Found ${#video_files[@]} video files"
    echo ""
    
    # Process each video
    for video_file in "${video_files[@]}"; do
        local filename=$(basename "$video_file")
        local name_without_ext="${filename%.*}"
        local output_file="$output_dir/$class_name/${name_without_ext}_processed.mp4"
        
        process_video "$video_file" "$output_file" "$class_name"
    done
}

# Function to create video metadata
create_metadata() {
    local processed_dir="$1"
    local metadata_file="$processed_dir/metadata.json"
    
    echo "📝 Creating metadata file: $metadata_file"
    
    # Create metadata JSON
    cat > "$metadata_file" << EOF
{
    "preprocessing_info": {
        "target_fps": $TARGET_FPS,
        "target_resolution": "$TARGET_RESOLUTION",
        "video_codec": "$VIDEO_CODEC",
        "audio_codec": "$AUDIO_CODEC",
        "processed_date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    },
    "classes": {}
}
EOF
    
    # Add class information
    for class_dir in "$processed_dir"/*; do
        if [ -d "$class_dir" ]; then
            local class_name=$(basename "$class_dir")
            local video_count=$(find "$class_dir" -name "*.mp4" | wc -l)
            
            # Update metadata with class info
            python3 -c "
import json
import sys

# Read existing metadata
with open('$metadata_file', 'r') as f:
    metadata = json.load(f)

# Add class information
metadata['classes']['$class_name'] = {
    'video_count': $video_count,
    'directory': '$class_name'
}

# Write updated metadata
with open('$metadata_file', 'w') as f:
    json.dump(metadata, f, indent=2)
"
        fi
    done
    
    echo "✅ Metadata created successfully"
}

# Function to validate processed videos
validate_videos() {
    local processed_dir="$1"
    
    echo "🔍 Validating processed videos..."
    
    local total_videos=0
    local valid_videos=0
    local invalid_videos=0
    
    # Check each video
    for video_file in $(find "$processed_dir" -name "*.mp4"); do
        total_videos=$((total_videos + 1))
        
        # Check if video is valid
        if ffprobe -v quiet -show_entries format=duration -of csv=p=0 "$video_file" > /dev/null 2>&1; then
            valid_videos=$((valid_videos + 1))
        else
            invalid_videos=$((invalid_videos + 1))
            echo "❌ Invalid video: $video_file"
        fi
    done
    
    echo "📊 Validation Results:"
    echo "  Total videos: $total_videos"
    echo "  Valid videos: $valid_videos"
    echo "  Invalid videos: $invalid_videos"
    echo ""
    
    if [ $invalid_videos -gt 0 ]; then
        echo "⚠️  Some videos failed validation. Check the logs above."
        return 1
    else
        echo "✅ All videos passed validation!"
        return 0
    fi
}

# Main execution
main() {
    echo "🚀 Starting Football Video Preprocessing"
    echo "📁 Raw videos directory: $RAW_VIDEOS_DIR"
    echo "📁 Processed videos directory: $PROCESSED_VIDEOS_DIR"
    echo "🎯 Target FPS: $TARGET_FPS"
    echo "📺 Target resolution: $TARGET_RESOLUTION"
    echo ""
    
    # Check if raw videos directory exists
    if [ ! -d "$RAW_VIDEOS_DIR" ]; then
        echo "❌ Raw videos directory not found: $RAW_VIDEOS_DIR"
        echo "💡 Run the download script first: ./01_data_collection/download_videos.sh"
        exit 1
    fi
    
    # Process each class directory
    for class_dir in "$RAW_VIDEOS_DIR"/*; do
        if [ -d "$class_dir" ]; then
            local class_name=$(basename "$class_dir")
            echo "🏷️  Processing class: $class_name"
            process_directory "$class_dir" "$PROCESSED_VIDEOS_DIR" "$class_name"
        fi
    done
    
    # Process other directories (channels, playlists)
    for other_dir in "$RAW_VIDEOS_DIR"/*; do
        if [ -d "$other_dir" ] && [ "$(basename "$other_dir")" != "channels" ] && [ "$(basename "$other_dir")" != "playlists" ]; then
            local dir_name=$(basename "$other_dir")
            echo "📁 Processing directory: $dir_name"
            process_directory "$other_dir" "$PROCESSED_VIDEOS_DIR" "$dir_name"
        fi
    done
    
    # Create metadata
    create_metadata "$PROCESSED_VIDEOS_DIR"
    
    # Validate processed videos
    validate_videos "$PROCESSED_VIDEOS_DIR"
    
    # Summary
    echo "📊 Preprocessing Summary:"
    local total_processed=$(find "$PROCESSED_VIDEOS_DIR" -name "*.mp4" | wc -l)
    echo "  Total processed videos: $total_processed"
    echo ""
    echo "✅ Preprocessing completed successfully!"
}

# Run main function
main "$@"