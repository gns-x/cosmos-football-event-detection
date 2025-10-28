'use client';

import { useState } from 'react';
import VideoPlayer from '@/components/VideoPlayer';
import PromptCard from '@/components/PromptCard';
import OutputSection from '@/components/OutputSection';

export default function Home() {
  const [userPrompt, setUserPrompt] = useState('Which worker picked up the dropped box?');
  const [systemPrompt, setSystemPrompt] = useState(`You are a professional football video analyst. Analyze the provided video and identify significant events.

<think>
- Watch the video carefully and identify all football events
- Look for goals, cards, shots, penalties, and other significant moments
- Note the timing and context of each event
- Consider player actions, referee decisions, and game flow
</think>

<answer>
Provide a structured analysis with:
1. Event type and description
2. Timestamp when the event occurred
3. Key details about the event
4. Any relevant context or implications
</answer>`);

  const [output] = useState({
    reasoning: [
      "Analyzed the video content frame by frame",
      "Identified multiple football events throughout the clip",
      "Noted the timing and context of each significant moment",
      "Considered player actions and referee decisions"
    ],
    response: "Based on my analysis of the video, I identified several key football events including goals, cards, and shots on target. The most significant events occurred at specific timestamps with clear visual indicators."
  });

  const [activeTab, setActiveTab] = useState<'preview' | 'json'>('preview');

  return (
    <main className="min-h-screen bg-[#0D1117] p-6">
      <div className="max-w-7xl mx-auto">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-white mb-2">Football Video Analysis</h1>
          <p className="text-[#8B949E]">AI-powered football event detection using Cosmos-Reason1-7B</p>
        </div>

        <div className="two-column-layout flex gap-6">
          {/* Left Column - Video Player */}
          <div className="video-column flex-1">
            <VideoPlayer />
          </div>

          {/* Right Column - Controls */}
          <div className="controls-column w-96 space-y-6">
            {/* User Prompt Card */}
            <PromptCard
              title="User Prompt"
              value={userPrompt}
              onChange={setUserPrompt}
              maxLength={250}
              placeholder="Enter your question about the video..."
            />

            {/* System Prompt Card */}
            <PromptCard
              title="System Prompt"
              value={systemPrompt}
              onChange={setSystemPrompt}
              maxLength={4000}
              placeholder="Enter system instructions..."
              rows={8}
            />

            {/* Output Section */}
            <OutputSection
              output={output}
              activeTab={activeTab}
              onTabChange={setActiveTab}
            />
          </div>
        </div>
      </div>
    </main>
  );
}