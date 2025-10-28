'use client';

interface OutputSectionProps {
  output: {
    reasoning: string[];
    response: string;
  };
  activeTab: 'preview' | 'json';
  onTabChange: (tab: 'preview' | 'json') => void;
}

export default function OutputSection({ output, activeTab, onTabChange }: OutputSectionProps) {
  const jsonOutput = {
    reasoning: output.reasoning,
    response: output.response,
    timestamp: new Date().toISOString(),
    model: "Cosmos-Reason1-7B",
    confidence: 0.92
  };

  return (
    <div className="nvidia-card">
      <h3 className="nvidia-card-header">Output</h3>
      
      {/* Tabs */}
      <div className="flex border-b border-[#30363D] mb-4">
        <button
          className={`nvidia-tab ${activeTab === 'preview' ? 'active' : ''}`}
          onClick={() => onTabChange('preview')}
        >
          Preview
        </button>
        <button
          className={`nvidia-tab ${activeTab === 'json' ? 'active' : ''}`}
          onClick={() => onTabChange('json')}
        >
          JSON
        </button>
      </div>

      {/* Content */}
      <div className="min-h-[300px]">
        {activeTab === 'preview' ? (
          <div className="space-y-4">
            <div>
              <h4 className="text-lg font-semibold text-white mb-3">Reasoning Complete</h4>
              <ul className="space-y-2">
                {output.reasoning.map((item, index) => (
                  <li key={index} className="flex items-start">
                    <span className="text-[#76B900] mr-2">•</span>
                    <span className="text-[#F0F6FC]">{item}</span>
                  </li>
                ))}
              </ul>
            </div>
            
            <div className="border-t border-[#30363D] pt-4">
              <h4 className="text-lg font-semibold text-white mb-3">Response</h4>
              <p className="text-[#F0F6FC] leading-relaxed">
                {output.response}
              </p>
            </div>
          </div>
        ) : (
          <div className="bg-[#0D1117] border border-[#30363D] rounded-lg p-4">
            <pre className="text-sm text-[#F0F6FC] overflow-x-auto">
              {JSON.stringify(jsonOutput, null, 2)}
            </pre>
          </div>
        )}
      </div>
    </div>
  );
}
