#!/bin/bash
# Setup Automated Video Downloader
# Configures automatic video downloading without manual commands

set -e

echo "🚀 Setting up Automated Video Downloader"
echo "=========================================="

# Configuration
PROJECT_DIR="/home/ubuntu/cosmos-football-event-detection"
SERVICE_NAME="cosmos_video_downloader"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

# Install required dependencies
echo "📦 Installing dependencies..."
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv ffmpeg yt-dlp

# Install Python packages
echo "🐍 Installing Python packages..."
pip3 install schedule requests

# Create project directory
echo "📁 Setting up project directory..."
sudo mkdir -p "$PROJECT_DIR"
sudo chown -R ubuntu:ubuntu "$PROJECT_DIR"

# Copy service file
echo "🔧 Setting up systemd service..."
sudo cp "$PROJECT_DIR/scripts/cosmos_video_downloader.service" "$SERVICE_FILE"

# Update service file with correct paths
sudo sed -i "s|/home/ubuntu/cosmos-football-event-detection|$PROJECT_DIR|g" "$SERVICE_FILE"

# Reload systemd
echo "🔄 Reloading systemd..."
sudo systemctl daemon-reload

# Enable service
echo "✅ Enabling service..."
sudo systemctl enable "$SERVICE_NAME"

# Start service
echo "🚀 Starting service..."
sudo systemctl start "$SERVICE_NAME"

# Check service status
echo "📊 Service status:"
sudo systemctl status "$SERVICE_NAME" --no-pager

# Create log directory
echo "📝 Setting up logging..."
mkdir -p "$PROJECT_DIR/logs"
chmod 755 "$PROJECT_DIR/logs"

# Create initial configuration
echo "⚙️  Creating initial configuration..."
cat > "$PROJECT_DIR/download_rules.json" << 'EOF'
{
  "download_rules": {
    "enabled": true,
    "schedule": "daily",
    "time": "02:00",
    "max_videos_per_channel": 10,
    "max_videos_per_class": 5,
    "m3u8_duration": 30,
    "auto_cleanup": true,
    "validation_enabled": true
  },
  "sources": {
    "channels": [
      "FIFA", "UEFA", "PremierLeague", "LaLiga", "SerieA",
      "Bundesliga", "Ligue1", "ChampionsLeague", "EuropaLeague", "WorldCup"
    ],
    "search_terms": [
      "penalty_shot", "goal", "red_card", "yellow_card",
      "corner_kick", "free_kick", "throw_in", "offside"
    ],
    "m3u8_streams": [
      "https://h2ddsukdqkoh6b.data.mediastore.us-east-1.amazonaws.com/ucl-walionpapl-aug13-eb643a9-archive/master/ucl-walionpapl-aug13-eb643a9-archive.m3u8"
    ]
  },
  "validation": {
    "min_duration": 5,
    "max_duration": 300,
    "min_file_size": 100000,
    "max_file_size": 500000000,
    "required_codecs": ["h264", "avc1"],
    "min_resolution": [320, 240],
    "max_resolution": [1920, 1080]
  },
  "cleanup": {
    "remove_invalid": true,
    "max_storage_gb": 50,
    "cleanup_old_videos_days": 30
  }
}
EOF

# Set permissions
chmod 644 "$PROJECT_DIR/download_rules.json"

# Create cron job for monitoring
echo "⏰ Setting up monitoring cron job..."
cat > "$PROJECT_DIR/scripts/monitor_downloader.sh" << 'EOF'
#!/bin/bash
# Monitor video downloader service

SERVICE_NAME="cosmos_video_downloader"
LOG_FILE="/home/ubuntu/cosmos-football-event-detection/logs/monitor.log"

# Check if service is running
if ! systemctl is-active --quiet "$SERVICE_NAME"; then
    echo "$(date): Service $SERVICE_NAME is not running, restarting..." >> "$LOG_FILE"
    sudo systemctl restart "$SERVICE_NAME"
fi

# Check disk space
DISK_USAGE=$(df /home/ubuntu/cosmos-football-event-detection | tail -1 | awk '{print $5}' | sed 's/%//')
if [ "$DISK_USAGE" -gt 80 ]; then
    echo "$(date): Disk usage is $DISK_USAGE%, cleaning up..." >> "$LOG_FILE"
    # Run cleanup
    cd /home/ubuntu/cosmos-football-event-detection
    python3 scripts/video_download_scheduler.py --once
fi
EOF

chmod +x "$PROJECT_DIR/scripts/monitor_downloader.sh"

# Add cron job
echo "📅 Adding monitoring cron job..."
(crontab -l 2>/dev/null; echo "*/5 * * * * $PROJECT_DIR/scripts/monitor_downloader.sh") | crontab -

# Create status check script
echo "📊 Creating status check script..."
cat > "$PROJECT_DIR/scripts/check_status.sh" << 'EOF'
#!/bin/bash
# Check video downloader status

echo "🔍 Cosmos Video Downloader Status"
echo "=================================="

# Service status
echo "📊 Service Status:"
systemctl status cosmos_video_downloader --no-pager

echo ""
echo "📁 Download Directory:"
ls -la 01_data_collection/raw_videos/

echo ""
echo "📊 Video Count:"
find 01_data_collection/raw_videos -name "*.mp4" | wc -l

echo ""
echo "💾 Disk Usage:"
df -h /home/ubuntu/cosmos-football-event-detection

echo ""
echo "📝 Recent Logs:"
tail -20 logs/scheduler_*.log
EOF

chmod +x "$PROJECT_DIR/scripts/check_status.sh"

# Create manual run script
echo "🎮 Creating manual run script..."
cat > "$PROJECT_DIR/scripts/run_download.sh" << 'EOF'
#!/bin/bash
# Manually run video download

echo "🚀 Running manual video download..."
cd /home/ubuntu/cosmos-football-event-detection
python3 scripts/video_download_scheduler.py --once
EOF

chmod +x "$PROJECT_DIR/scripts/run_download.sh"

# Create M3U8 discovery script
echo "🔍 Creating M3U8 discovery script..."
cat > "$PROJECT_DIR/scripts/discover_m3u8_streams.sh" << 'EOF'
#!/bin/bash
# Discover M3U8 streams

echo "🔍 Discovering M3U8 streams..."
cd /home/ubuntu/cosmos-football-event-detection
python3 scripts/m3u8_stream_discovery.py --output discovered_m3u8_streams.json
echo "✅ M3U8 stream discovery completed"
EOF

chmod +x "$PROJECT_DIR/scripts/discover_m3u8_streams.sh"

# Create stop script
echo "⏹️  Creating stop script..."
cat > "$PROJECT_DIR/scripts/stop_downloader.sh" << 'EOF'
#!/bin/bash
# Stop video downloader

echo "⏹️  Stopping video downloader..."
sudo systemctl stop cosmos_video_downloader
sudo systemctl disable cosmos_video_downloader
echo "✅ Video downloader stopped"
EOF

chmod +x "$PROJECT_DIR/scripts/stop_downloader.sh"

# Create start script
echo "▶️  Creating start script..."
cat > "$PROJECT_DIR/scripts/start_downloader.sh" << 'EOF'
#!/bin/bash
# Start video downloader

echo "▶️  Starting video downloader..."
sudo systemctl start cosmos_video_downloader
sudo systemctl enable cosmos_video_downloader
echo "✅ Video downloader started"
EOF

chmod +x "$PROJECT_DIR/scripts/start_downloader.sh"

# Final setup
echo "🔧 Final setup..."
cd "$PROJECT_DIR"

# Test the automated downloader
echo "🧪 Testing automated downloader..."
python3 scripts/automated_video_downloader.py --help

echo ""
echo "✅ Automated Video Downloader Setup Complete!"
echo "============================================="
echo ""
echo "📊 Service Status:"
sudo systemctl status "$SERVICE_NAME" --no-pager
echo ""
echo "🎮 Available Commands:"
echo "  ./scripts/check_status.sh     # Check status"
echo "  ./scripts/run_download.sh    # Run download manually"
echo "  ./scripts/discover_m3u8_streams.sh # Discover M3U8 streams"
echo "  ./scripts/start_downloader.sh # Start service"
echo "  ./scripts/stop_downloader.sh  # Stop service"
echo ""
echo "📁 Configuration:"
echo "  $PROJECT_DIR/download_rules.json"
echo ""
echo "📝 Logs:"
echo "  $PROJECT_DIR/logs/"
echo ""
echo "🚀 The system will automatically download videos daily at 2:00 AM"
echo "✅ Setup completed successfully!"
