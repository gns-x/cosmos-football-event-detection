#!/bin/bash
# Test video path resolution for evaluation

echo "🔍 Testing video path resolution for evaluation..."
echo "================================================"

# Check current directory
echo "Current directory: $(pwd)"

# Check if we're in the right location
if [ -d "06_evaluation" ]; then
    echo "✅ Found 06_evaluation directory"
    cd 06_evaluation
    echo "Changed to: $(pwd)"
else
    echo "❌ 06_evaluation directory not found"
    exit 1
fi

# Test video paths from evaluation perspective
echo ""
echo "🔍 Testing video paths from evaluation directory..."

test_paths=(
    "01_data_collection/raw_videos/yellow_card/yellow_card_Craziest Yellow Cards in Football.mp4"
    "../01_data_collection/raw_videos/yellow_card/yellow_card_Craziest Yellow Cards in Football.mp4"
    "../../01_data_collection/raw_videos/yellow_card/yellow_card_Craziest Yellow Cards in Football.mp4"
)

for path in "${test_paths[@]}"; do
    if [ -f "$path" ]; then
        echo "✅ Found: $path"
    else
        echo "❌ Not found: $path"
    fi
done

echo ""
echo "🔍 Checking if any videos exist in data collection..."
if [ -d "../01_data_collection/raw_videos" ]; then
    echo "✅ Data collection directory exists"
    echo "Videos found:"
    find ../01_data_collection/raw_videos -name "*.mp4" | head -5
else
    echo "❌ Data collection directory not found"
fi
