import { useState, useEffect } from 'react';
import { cosmosAPI } from '../services/cosmosAPI';

export default function CreateDataset() {
  const [status, setStatus] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<{ success: boolean; message: string; records: number } | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadStatus();
  }, []);

  const loadStatus = async () => {
    try {
      const s = await cosmosAPI.getPipelineStatus();
      setStatus(s);
    } catch (err: any) {
      setError(err.message);
    }
  };

  const handleCreateDataset = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    
    try {
      const res = await cosmosAPI.createDataset();
      setResult(res);
      await loadStatus(); // Refresh status
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-[1200px] mx-auto p-6 space-y-6">
      <div className="bg-[#121212] border-2 border-transparent rounded-lg shadow-lg p-6 relative group hover:border-[#76B900] transition-all">
        <h1 className="text-2xl font-bold text-[#76B900] mb-4">Step 2: Create Dataset</h1>
        <p className="text-gray-300 mb-6">
          This step combines video clips from <code className="bg-[#1a1a1a] px-2 py-1 rounded">01_clips/</code> with their
          annotations from <code className="bg-[#1a1a1a] px-2 py-1 rounded">02_annotations/</code> to create a training dataset
          in LLaVA format. The dataset is saved to <code className="bg-[#1a1a1a] px-2 py-1 rounded">03_dataset/train_dataset.jsonl</code>.
        </p>

        {status && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <div className="bg-[#1a1a1a] rounded-lg p-4 border border-gray-700">
              <div className="text-xs text-gray-400 mb-1">Clips</div>
              <div className="text-2xl font-bold text-[#76B900]">{status.clips_count}</div>
            </div>
            <div className="bg-[#1a1a1a] rounded-lg p-4 border border-gray-700">
              <div className="text-xs text-gray-400 mb-1">Annotations</div>
              <div className="text-2xl font-bold text-[#76B900]">{status.annotations_count}</div>
            </div>
            <div className="bg-[#1a1a1a] rounded-lg p-4 border border-gray-700">
              <div className="text-xs text-gray-400 mb-1">Dataset Records</div>
              <div className="text-2xl font-bold text-[#76B900]">{status.dataset_count}</div>
            </div>
            <div className="bg-[#1a1a1a] rounded-lg p-4 border border-gray-700">
              <div className="text-xs text-gray-400 mb-1">Ready</div>
              <div className={`text-sm font-medium ${status.has_annotations ? 'text-green-400' : 'text-yellow-400'}`}>
                {status.has_annotations ? 'Yes' : 'No'}
              </div>
            </div>
          </div>
        )}

        <button
          onClick={handleCreateDataset}
          disabled={loading || !status?.has_annotations}
          className="w-full px-6 py-4 bg-gradient-to-r from-[#76B900] to-[#87ca00] hover:from-[#87ca00] hover:to-[#76B900] text-black font-semibold rounded-lg transition-all duration-300 hover:scale-105 hover:shadow-[0_0_20px_rgba(118,185,0,0.4)] active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100"
        >
          {loading ? (
            <div className="flex items-center justify-center gap-2">
              <div className="w-5 h-5 border-2 border-black border-t-transparent rounded-full animate-spin"></div>
              Creating Dataset...
            </div>
          ) : (
            'Create Dataset'
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
              {result.success ? '✓ Success' : '✗ Failed'}
            </div>
            <div className="text-gray-300 text-sm">{result.message}</div>
            {result.success && (
              <div className="mt-2 text-sm text-gray-400">
                Created {result.records} training records
              </div>
            )}
          </div>
        )}

        {!status?.has_annotations && (
          <div className="mt-4 p-4 bg-yellow-900/20 border border-yellow-700 rounded-lg text-yellow-300">
            <strong>Note:</strong> No annotations found. Please run "Generate Annotations" first.
          </div>
        )}
      </div>
    </div>
  );
}
