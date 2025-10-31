import { useState, useEffect } from 'react';
import { cosmosAPI } from '../services/cosmosAPI';

export default function Training() {
  const [status, setStatus] = useState<any>(null);
  const [trainingStatus, setTrainingStatus] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<{ success: boolean; message: string; job_id: string } | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadStatus();
    loadTrainingStatus();
    const interval = setInterval(loadTrainingStatus, 3000); // Poll every 3s
    return () => clearInterval(interval);
  }, []);

  const loadStatus = async () => {
    try {
      const s = await cosmosAPI.getPipelineStatus();
      setStatus(s);
    } catch (err: any) {
      setError(err.message);
    }
  };

  const loadTrainingStatus = async () => {
    try {
      const ts = await cosmosAPI.getTrainingStatus();
      setTrainingStatus(ts);
    } catch (err: any) {
      // Ignore errors for status polling
    }
  };

  const handleStartTraining = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    
    try {
      const res = await cosmosAPI.startTraining();
      setResult(res);
      await loadTrainingStatus();
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-[1200px] mx-auto p-6 space-y-6">
      <div className="bg-[#121212] border-2 border-transparent rounded-lg shadow-lg p-6 relative group hover:border-[#76B900] transition-all">
        <h1 className="text-2xl font-bold text-[#76B900] mb-4">Step 3: Train Model</h1>
        <p className="text-gray-300 mb-6">
          This step fine-tunes the Cosmos model using LoRA (Low-Rank Adaptation) on the generated dataset.
          The fine-tuned adapters are saved to <code className="bg-[#1a1a1a] px-2 py-1 rounded">04_model_output/final_checkpoint/</code>.
        </p>

        {status && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <div className="bg-[#1a1a1a] rounded-lg p-4 border border-gray-700">
              <div className="text-xs text-gray-400 mb-1">Dataset Records</div>
              <div className="text-2xl font-bold text-[#76B900]">{status.dataset_count}</div>
            </div>
            <div className="bg-[#1a1a1a] rounded-lg p-4 border border-gray-700">
              <div className="text-xs text-gray-400 mb-1">Model Status</div>
              <div className={`text-sm font-medium ${status.has_model ? 'text-green-400' : 'text-red-400'}`}>
                {status.has_model ? 'Available' : 'Not Found'}
              </div>
            </div>
            <div className="bg-[#1a1a1a] rounded-lg p-4 border border-gray-700">
              <div className="text-xs text-gray-400 mb-1">Trained</div>
              <div className={`text-sm font-medium ${status.has_trained ? 'text-green-400' : 'text-yellow-400'}`}>
                {status.has_trained ? 'Yes' : 'No'}
              </div>
            </div>
            <div className="bg-[#1a1a1a] rounded-lg p-4 border border-gray-700">
              <div className="text-xs text-gray-400 mb-1">Ready</div>
              <div className={`text-sm font-medium ${status.has_dataset && status.has_model ? 'text-green-400' : 'text-yellow-400'}`}>
                {status.has_dataset && status.has_model ? 'Yes' : 'No'}
              </div>
            </div>
          </div>
        )}

        {trainingStatus && (
          <div className="mb-6 p-4 bg-[#1a1a1a] rounded-lg border border-gray-700">
            <div className="flex items-center justify-between mb-2">
              <div className="text-sm font-medium text-gray-300">Training Status</div>
              <div className={`text-sm font-medium ${trainingStatus.running ? 'text-green-400' : 'text-gray-400'}`}>
                {trainingStatus.running ? 'Running' : 'Idle'}
              </div>
            </div>
            {trainingStatus.running && (
              <>
                <div className="w-full bg-gray-800 rounded-full h-2 mb-2">
                  <div
                    className="bg-[#76B900] h-2 rounded-full transition-all"
                    style={{ width: `${trainingStatus.progress}%` }}
                  ></div>
                </div>
                <div className="text-xs text-gray-400">{trainingStatus.current_step}</div>
                {trainingStatus.logs && trainingStatus.logs.length > 0 && (
                  <div className="mt-3 text-xs font-mono text-gray-500 max-h-32 overflow-y-auto">
                    {trainingStatus.logs.slice(-5).map((log: string, i: number) => (
                      <div key={i}>{log}</div>
                    ))}
                  </div>
                )}
              </>
            )}
          </div>
        )}

        <button
          onClick={handleStartTraining}
          disabled={loading || !status?.has_dataset || !status?.has_model || trainingStatus?.running}
          className="w-full px-6 py-4 bg-gradient-to-r from-[#76B900] to-[#87ca00] hover:from-[#87ca00] hover:to-[#76B900] text-black font-semibold rounded-lg transition-all duration-300 hover:scale-105 hover:shadow-[0_0_20px_rgba(118,185,0,0.4)] active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100"
        >
          {loading ? (
            <div className="flex items-center justify-center gap-2">
              <div className="w-5 h-5 border-2 border-black border-t-transparent rounded-full animate-spin"></div>
              Starting Training...
            </div>
          ) : (
            'Start Training'
          )}
        </button>

        {error && (
          <div className="mt-4 p-4 bg-red-900/20 border border-red-700 rounded-lg text-red-300">
            <strong>Error:</strong> {error}
          </div>
        )}

        {result && (
          <div className={`mt-4 p-4 rounded-lg ${result.success ? 'bg-green-900/20 border border-green-700' : 'bg-red-900/20 border border-red-700'}`}>
            <div className={`font-semibold mb-2 ${result.success ? 'text-green-300' : 'text-red-300'}`}>
              {result.success ? '✓ Training Started' : '✗ Failed'}
            </div>
            <div className="text-gray-300 text-sm">{result.message}</div>
          </div>
        )}

        {!status?.has_dataset && (
          <div className="mt-4 p-4 bg-yellow-900/20 border border-yellow-700 rounded-lg text-yellow-300">
            <strong>Note:</strong> No dataset found. Please run "Create Dataset" first.
          </div>
        )}

        <div className="mt-4 p-4 bg-blue-900/20 border border-blue-700 rounded-lg text-blue-300 text-sm">
          <strong>Note:</strong> Training runs the script at <code className="bg-[#1a1a1a] px-1 py-0.5 rounded">05_scripts/train.py</code>.
          For full training, run it manually on a machine with GPU access.
        </div>
      </div>
    </div>
  );
}
