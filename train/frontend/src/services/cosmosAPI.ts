// API service for connecting frontend to Cosmos backend (optional)
// Default to VM IP if no env var provided
const API_BASE_URL = (import.meta as unknown as { env?: Record<string, string> }).env?.VITE_API_BASE_URL || 'http://localhost:8000';

export interface VideoAnalysisRequest {
  prompt: string;
  systemPrompt?: string;
  videoFile?: File;
}

export interface AnalysisResponse {
  reasoning: string[];
  answer: string;
  confidence: number;
  timestamp: string;
  actor: string;
  events?: EventData[];
  summary?: Record<string, unknown>;
}

export interface EventData {
  event_type: string;
  start_time: string;
  end_time: string;
  player_jersey: string;
  team: string;
  jersey_color: string;
  description?: string;
  assist_player?: string;
  goal_type?: string;
  outcome?: string;
  goalkeeper_jersey?: string;
  goalkeeper_team?: string;
  goalkeeper_color?: string;
  reason?: string;
  referee_action?: string;
  foul_type?: string;
}

class CosmosAPIService {
  private baseURL: string;

  constructor(baseURL: string = API_BASE_URL) {
    this.baseURL = baseURL;
  }

  // Get detailed health status (resilient, no-throw)
  async getHealthStatus(): Promise<{
    status: string;
    model_loaded: boolean;
    device: string;
    model_name: string;
  }> {
    try {
      if (!this.baseURL) {
        return { status: 'offline', model_loaded: false, device: 'none', model_name: 'none' };
      }
      const response = await fetch(`${this.baseURL}/health`);
      const data = await response.json();
      return {
        status: data.status,
        model_loaded: data.nim_ready,
        device: 'nvidia_nim',
        model_name: data.model
      };
    } catch {
      return { status: 'offline', model_loaded: false, device: 'none', model_name: 'none' };
    }
  }

  // Analyze video with file upload
  async analyzeVideo(request: VideoAnalysisRequest): Promise<AnalysisResponse> {
    try {
      if (!this.baseURL) {
        throw new Error('Backend is disabled');
      }
      const formData = new FormData();
      formData.append('prompt', request.prompt);
      
      if (request.systemPrompt) {
        formData.append('system_prompt', request.systemPrompt);
      }
      
      if (request.videoFile) {
        formData.append('file', request.videoFile);
      }

      const response = await fetch(`${this.baseURL}/analyze`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Analysis failed');
      }

      return await response.json();
    } catch {
      throw new Error('Backend not available');
    }
  }

  // Check if backend is ready
  async isReady(): Promise<boolean> {
    const health = await this.getHealthStatus();
    return health.status === 'healthy' && health.model_loaded;
  }

  // Pipeline endpoints
  async getPipelineStatus(): Promise<{
    has_clips: boolean;
    has_annotations: boolean;
    has_dataset: boolean;
    has_model: boolean;
    has_trained: boolean;
    clips_count: number;
    annotations_count: number;
    dataset_count: number;
  }> {
    const response = await fetch(`${this.baseURL}/pipeline/status`);
    if (!response.ok) throw new Error('Failed to get pipeline status');
    return await response.json();
  }

  async generateAnnotations(): Promise<{
    success: boolean;
    message: string;
    processed: number;
    generated: number;
  }> {
    const response = await fetch(`${this.baseURL}/pipeline/generate-annotations`, {
      method: 'POST',
    });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to generate annotations');
    }
    return await response.json();
  }

  async createDataset(): Promise<{
    success: boolean;
    message: string;
    records: number;
  }> {
    const response = await fetch(`${this.baseURL}/pipeline/create-dataset`, {
      method: 'POST',
    });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to create dataset');
    }
    return await response.json();
  }

  async startTraining(): Promise<{
    success: boolean;
    message: string;
    job_id: string;
  }> {
    const response = await fetch(`${this.baseURL}/pipeline/train`, {
      method: 'POST',
    });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to start training');
    }
    return await response.json();
  }

  async getTrainingStatus(): Promise<{
    running: boolean;
    progress: number;
    current_step: string;
    logs: string[];
  }> {
    const response = await fetch(`${this.baseURL}/pipeline/train/status`);
    if (!response.ok) throw new Error('Failed to get training status');
    return await response.json();
  }

  async testInference(videoFile: File, opts?: { prompt?: string; systemPrompt?: string }): Promise<{
    success: boolean;
    events: any[];
    raw_output: string;
    error?: string;
  }> {
    const formData = new FormData();
    formData.append('file', videoFile);
    if (opts?.prompt) formData.append('prompt', opts.prompt);
    if (opts?.systemPrompt) formData.append('system_prompt', opts.systemPrompt);
    const response = await fetch(`${this.baseURL}/pipeline/test-inference`, {
      method: 'POST',
      body: formData,
    });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to run inference');
    }
    return await response.json();
  }
}

// Export singleton instance
export const cosmosAPI = new CosmosAPIService();

// Export class for custom instances
export default CosmosAPIService;
