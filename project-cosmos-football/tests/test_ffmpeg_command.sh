#!/bin/bash

# Test script to demonstrate the exact ffmpeg command from Cosmos-Reason1-7B model card
# This shows the exact format: ffmpeg -i input.mp4 -r 4 output.mp4

echo "🧪 Testing ffmpeg command format from Cosmos-Reason1-7B model card"
echo ""

# Create a test input video (1 second, 30fps)
echo "📹 Creating test input video (30 FPS)..."
ffmpeg -y -f lavfi -i "testsrc=duration=1:size=720x480:rate=30" -c:v libx264 test_input_30fps.mp4

# Test the exact command format from the model card
echo "🔄 Testing exact command: ffmpeg -i input.mp4 -r 4 output.mp4"
ffmpeg -i test_input_30fps.mp4 -r 4 test_output_4fps.mp4

# Verify the output
echo "✅ Verifying output video..."
ORIGINAL_FPS=$(ffprobe -v quiet -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 test_input_30fps.mp4)
OUTPUT_FPS=$(ffprobe -v quiet -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 test_output_4fps.mp4)

echo "📊 Results:"
echo "  Original FPS: $ORIGINAL_FPS"
echo "  Output FPS: $OUTPUT_FPS"
echo "  Target FPS: 4"
echo ""

if [ "$OUTPUT_FPS" = "4/1" ] || [ "$OUTPUT_FPS" = "4" ]; then
    echo "✅ SUCCESS: Video correctly resampled to 4 FPS!"
else
    echo "❌ FAILED: Video not resampled to 4 FPS"
fi

# Cleanup
rm -f test_input_30fps.mp4 test_output_4fps.mp4

echo ""
echo "🎯 This demonstrates the exact ffmpeg command format required by Cosmos-Reason1-7B:"
echo "   ffmpeg -i /raw_videos/goal_01.mp4 -r 4 /processed_videos/goal_01.mp4"
