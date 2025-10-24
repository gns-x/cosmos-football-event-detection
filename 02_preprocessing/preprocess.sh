#!/bin/bash

# Script to preprocess videos using ffmpeg
# Resamples videos to 4 FPS and standardizes format

echo "Starting video preprocessing..."

# Create processed_videos directory if it doesn't exist
mkdir -p processed_videos

# Process all videos in raw_videos directory
for video in ../01_data_collection/raw_videos/*.mp4; do
    if [ -f "$video" ]; then
        filename=$(basename "$video" .mp4)
        echo "Processing: $filename"
        
        # Resample to 4 FPS and convert to standard format
        ffmpeg -i "$video" \
            -vf "fps=4" \
            -c:v libx264 \
            -c:a aac \
            -y \
            "processed_videos/${filename}_4fps.mp4"
    fi
done

echo "Video preprocessing completed. Check processed_videos/ directory for processed files."
