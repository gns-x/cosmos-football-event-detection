'use client';

import { useState, useRef, useEffect } from 'react';

export default function VideoPlayer() {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(30); // 30 seconds as specified
  const videoRef = useRef<HTMLVideoElement>(null);

  const togglePlay = () => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause();
      } else {
        videoRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  const handleTimeUpdate = () => {
    if (videoRef.current) {
      setCurrentTime(videoRef.current.currentTime);
    }
  };

  const handleLoadedMetadata = () => {
    if (videoRef.current) {
      setDuration(videoRef.current.duration);
    }
  };

  const formatTime = (time: number) => {
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  };

  const progressPercentage = (currentTime / duration) * 100;

  return (
    <div className="nvidia-card">
      <div className="video-player">
        {/* Video Element */}
        <div className="w-full h-64 bg-[#161B22] rounded-t-lg flex items-center justify-center border border-[#30363D]">
          <div className="text-center">
            <div className="w-16 h-16 mx-auto mb-4 bg-[#76B900] rounded-full flex items-center justify-center">
              <svg width="32" height="32" viewBox="0 0 24 24" fill="white">
                <path d="M8 5v14l11-7z"/>
              </svg>
            </div>
            <h3 className="text-lg font-semibold text-white mb-2">Football Video Player</h3>
            <p className="text-[#8B949E] text-sm">Upload a video file to begin analysis</p>
            <button className="nvidia-button mt-4">
              Upload Video
            </button>
          </div>
        </div>

        {/* Timeline */}
        <div className="px-4">
          <div className="video-timeline">
            <div 
              className="video-progress"
              style={{ width: `${progressPercentage}%` }}
            />
          </div>
        </div>

        {/* Controls */}
        <div className="video-controls">
          <button 
            className="play-button"
            onClick={togglePlay}
            aria-label={isPlaying ? 'Pause' : 'Play'}
          >
            {isPlaying ? (
              <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                <path d="M6 4h4v16H6V4zm8 0h4v16h-4V4z"/>
              </svg>
            ) : (
              <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                <path d="M8 5v14l11-7z"/>
              </svg>
            )}
          </button>

          <div className="time-display">
            {formatTime(currentTime)} / {formatTime(duration)}
          </div>

          <div className="flex-1" />

          <div className="text-sm text-[#8B949E]">
            Football Analysis Video
          </div>
        </div>
      </div>
    </div>
  );
}
