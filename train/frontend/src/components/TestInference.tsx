import { useState, useRef } from 'react';
import { cosmosAPI } from '../services/cosmosAPI';

export default function TestInference() {
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<{ success: boolean; events: any[]; raw_output: string; error?: string } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const [userPrompt, setUserPrompt] = useState("Give me all the goals in this video");
  const [systemPrompt, setSystemPrompt] = useState(
    `You are a professional football analyst. Analyze the video content and provide detailed insights based on what you observe. Focus on the user's specific request and provide accurate analysis.`
  );

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && file.type.startsWith('video/')) {
      setUploadedFile(file);
      if (videoUrl) URL.revokeObjectURL(videoUrl);
      const url = URL.createObjectURL(file);
      setVideoUrl(url);
      setResult(null);
      setError(null);
    }
  };

  const handleTestInference = async () => {
    if (!uploadedFile) {
      setError('Please upload a video first');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await cosmosAPI.testInference(uploadedFile, { prompt: userPrompt, systemPrompt });
      setResult(res);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-[1200px] mx-auto p-6 space-y-6">
      <div className="bg-[#121212] border-2 border-transparent rounded-lg shadow-lg p-6 relative group hover:border-[#76B900] transition-all">
        <h1 className="text-2xl font-bold text-[#76B900] mb-4">Step 4: Test Inference (Trained model)</h1>
        <p className="text-gray-300 mb-6">
          Test the fine-tuned adapters on a video clip. This uses the <b>Trained model</b> (base + LoRA).
        </p>

        <div className="space-y-4">
          <div className="bg-[#1a1a1a] rounded-lg p-4 border border-gray-700">
            <div className="flex items-center gap-4">
              <input
                ref={fileInputRef}
                type="file"
                accept="video/*"
                onChange={handleFileUpload}
                className="hidden"
              />
              <button
                onClick={() => fileInputRef.current?.click()}
                className="px-4 py-2 bg-[#76B900] text-black rounded-lg hover:bg-[#87ca00] transition"
              >
                Upload Video
              </button>
              {uploadedFile && (
                <div className="text-sm text-gray-300">
                  {uploadedFile.name}
                </div>
              )}
            </div>
          </div>

          {videoUrl && (
            <div className="bg-[#1a1a1a] rounded-lg p-4 border border-gray-700">
              <video ref={videoRef} src={videoUrl} controls className="w-full max-h-96 rounded" />
            </div>
          )}

          {/* Prompts (same UX as Analyze) */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div className="bg-[#1a1a1a] rounded-lg p-4 border border-gray-700">
              <h2 className="text-sm font-semibold text-[#76B900] mb-2">System Prompt</h2>
              <textarea
                value={systemPrompt}
                onChange={(e) => setSystemPrompt(e.target.value)}
                rows={5}
                className="w-full bg-[#0f0f0f] border border-gray-700 rounded px-3 py-2 text-sm"
              />
            </div>
            <div className="bg-[#1a1a1a] rounded-lg p-4 border border-gray-700">
              <h2 className="text-sm font-semibold text-[#76B900] mb-2">User Prompt</h2>
              <textarea
                value={userPrompt}
                onChange={(e) => setUserPrompt(e.target.value)}
                rows={5}
                className="w-full bg-[#0f0f0f] border border-gray-700 rounded px-3 py-2 text-sm"
              />
            </div>
          </div>

          <button
            onClick={handleTestInference}
            disabled={loading || !uploadedFile}
            className="w-full px-6 py-4 bg-gradient-to-r from-[#76B900] to-[#87ca00] hover:from-[#87ca00] hover:to-[#76B900] text-black font-semibold rounded-lg transition-all duration-300 hover:scale-105 hover:shadow-[0_0_20px_rgba(118,185,0,0.4)] active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100"
          >
            {loading ? (
              <div className="flex items-center justify-center gap-2">
                <div className="w-5 h-5 border-2 border-black border-t-transparent rounded-full animate-spin"></div>
                Running Inference...
              </div>
            ) : (
              'Run Inference'
            )}
          </button>

          {error && (
            <div className="p-4 bg-red-900/20 border border-red-700 rounded-lg text-red-300">
              <strong>Error:</strong> {error}
            </div>
          )}

          {result && (
            <div className="space-y-4">
              {result.error ? (
                <div className="p-4 bg-red-900/20 border border-red-700 rounded-lg text-red-300">
                  <strong>Error:</strong> {result.error}
                </div>
              ) : (
                <>
                  <div className="p-4 bg-green-900/20 border border-green-700 rounded-lg">
                    <div className="font-semibold text-green-300 mb-2">✓ Inference Complete (Trained model)</div>
                    <div className="text-sm text-gray-300">Detected {result.events.length} event(s)</div>
                  </div>

                  {result.events.length > 0 && (
                    <div className="bg-[#1a1a1a] rounded-lg p-4 border border-gray-700">
                      <h3 className="text-lg font-semibold text-[#76B900] mb-3">Detected Events</h3>
                      <div className="space-y-2">
                        {result.events.map((event: any, i: number) => (
                          <div key={i} className="p-3 bg-[#0f0f0f] rounded border border-gray-800">
                            <div className="flex items-center gap-3 mb-2">
                              <span className="px-2 py-1 bg-[#76B900] text-black text-xs font-semibold rounded">
                                {event.event || event.event_type || 'Event'}
                              </span>
                              {(event.start_time || event.startTime) && (
                                <span className="text-sm text-gray-400">
                                  {(event.start_time || event.startTime)} - {(event.end_time || event.endTime || '--:--')}
                                </span>
                              )}
                            </div>
                            {event.description && (
                              <div className="text-sm text-gray-300">{event.description}</div>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {result.raw_output && (
                    <details className="bg-[#1a1a1a] rounded-lg p-4 border border-gray-700">
                      <summary className="cursor-pointer text-sm font-medium text-gray-300 mb-2">
                        Raw Output
                      </summary>
                      <pre className="text-xs text-gray-400 overflow-x-auto mt-2">
                        {result.raw_output}
                      </pre>
                    </details>
                  )}
                </>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
