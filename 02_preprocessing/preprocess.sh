#!/bin/bash

# Football Video Preprocessing Script - Updated for Azure A100
# Processes raw videos to 4 FPS as required by Cosmos-Reason1-7B
# Handles incomplete downloads and special characters

set -e

# Configuration
RAW_VIDEOS_DIR="../01_data_collection/raw_videos"
PROCESSED_VIDEOS_DIR="../02_preprocessing/processed_videos"
TARGET_FPS=4
TARGET_RESOLUTION="720x480"
VIDEO_CODEC="libx264"
AUDIO_CODEC="aac"

# Create processed videos directory
mkdir -p "$PROCESSED_VIDEOS_DIR"

# Function to clean filename
clean_filename() {
    local filename="$1"
    # Remove special characters and replace with underscores
    echo "$filename" | sed 's/[^a-zA-Z0-9._-]/_/g' | sed 's/__*/_/g'
}

# Function to check if video is valid
is_valid_video() {
    local video_file="$1"
    
    # Check if file exists and has content
    if [ ! -f "$video_file" ] || [ ! -s "$video_file" ]; then
        return 1
    fi
    
    # Check if it's a valid video file
    if ffprobe -v quiet -show_entries format=duration -of csv=p=0 "$video_file" > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Function to process a single video
process_video() {
    local input_file="$1"
    local output_file="$2"
    local class_name="$3"
    
    echo "🎬 Processing: $(basename "$input_file")"
    
    # Extract video info
    local duration=$(ffprobe -v quiet -show_entries format=duration -of csv=p=0 "$input_file" 2>/dev/null || echo "0")
    local fps=$(ffprobe -v quiet -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 "$input_file" 2>/dev/null || echo "0/1")
    
    echo "  📊 Original: ${duration}s, ${fps} FPS"
    
    # Process video to 4 FPS
    if ffmpeg -i "$input_file" \
        -r 4 \
        -vf "scale=$TARGET_RESOLUTION" \
        -c:v "$VIDEO_CODEC" \
        -c:a "$AUDIO_CODEC" \
        -preset fast \
        -crf 23 \
        -y \
        "$output_file" 2>/dev/null; then
        
        # Verify output
        local new_duration=$(ffprobe -v quiet -show_entries format=duration -of csv=p=0 "$output_file" 2>/dev/null || echo "0")
        local new_fps=$(ffprobe -v quiet -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 "$output_file" 2>/dev/null || echo "0/1")
        
        echo "  ✅ Processed: ${new_duration}s, ${new_fps} FPS"
        echo "  💾 Saved to: $output_file"
        return 0
    else
        echo "  ❌ Failed to process: $input_file"
        return 1
    fi
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
    
    # Find all valid video files
    local video_files=()
    for video_file in "$input_dir"/*.mp4; do
        if [ -f "$video_file" ] && is_valid_video "$video_file"; then
            video_files+=("$video_file")
        fi
    done
    
    if [ ${#video_files[@]} -eq 0 ]; then
        echo "⚠️  No valid video files found in $input_dir"
        return
    fi
    
    echo "📹 Found ${#video_files[@]} valid video files"
    echo ""
    
    # Process each video
    local processed_count=0
    for video_file in "${video_files[@]}"; do
        local filename=$(basename "$video_file")
        local name_without_ext="${filename%.*}"
        local clean_name=$(clean_filename "$name_without_ext")
        local output_file="$output_dir/$class_name/${clean_name}_processed.mp4"
        
        if process_video "$video_file" "$output_file" "$class_name"; then
            processed_count=$((processed_count + 1))
        fi
        echo ""
    done
    
    echo "✅ Processed $processed_count out of ${#video_files[@]} videos for $class_name"
    echo ""
}

# Function to clean up incomplete downloads
cleanup_incomplete_downloads() {
    echo "🧹 Cleaning up incomplete downloads..."
    
    # Remove incomplete files
    find "$RAW_VIDEOS_DIR" -name "*.part" -delete
    find "$RAW_VIDEOS_DIR" -name "*.ytdl" -delete
    find "$RAW_VIDEOS_DIR" -name "*.tmp" -delete
    
    # Remove empty files
    find "$RAW_VIDEOS_DIR" -name "*.mp4" -size 0 -delete
    
    echo "✅ Cleanup completed"
    echo ""
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
        if is_valid_video "$video_file"; then
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
    echo "🚀 Starting Football Video Preprocessing (Updated for Azure A100)"
    echo "📁 Raw videos directory: $RAW_VIDEOS_DIR"
    echo "📁 Processed videos directory: $PROCESSED_VIDEOS_DIR"
    echo "🎯 Target FPS: $TARGET_FPS"
    echo "📺 Target resolution: $TARGET_RESOLUTION"
    echo ""
    
    # Check if raw videos directory exists
    if [ ! -d "$RAW_VIDEOS_DIR" ]; then
        echo "❌ Raw videos directory not found: $RAW_VIDEOS_DIR"
        echo "💡 Run the download script first: make download-videos"
        exit 1
    fi
    
    # Clean up incomplete downloads
    cleanup_incomplete_downloads
    
    # Process each class directory
    for class_dir in "$RAW_VIDEOS_DIR"/*; do
        if [ -d "$class_dir" ]; then
            local class_name=$(basename "$class_dir")
            echo "🏷️  Processing class: $class_name"
            process_directory "$class_dir" "$PROCESSED_VIDEOS_DIR" "$class_name"
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