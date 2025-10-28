// API service for connecting frontend to Cosmos backend
const API_BASE_URL = 'http://localhost:8000';

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

export interface ModelInfo {
  model_name: string;
  device: string;
  loaded: boolean;
  description: string;
}

class CosmosAPIService {
  private baseURL: string;

  constructor(baseURL: string = API_BASE_URL) {
    this.baseURL = baseURL;
  }

  // Health check
  async healthCheck(): Promise<{ message: string; model_loaded: boolean }> {
    try {
      const response = await fetch(`${this.baseURL}/`);
      return await response.json();
    } catch (error) {
      console.error('Health check failed:', error);
      throw new Error('Backend API is not available');
    }
  }

  // Get detailed health status
  async getHealthStatus(): Promise<{
    status: string;
    model_loaded: boolean;
    device: string;
    model_name: string;
  }> {
    try {
      const response = await fetch(`${this.baseURL}/health`);
      return await response.json();
    } catch (error) {
      console.error('Health status check failed:', error);
      throw new Error('Failed to get health status');
    }
  }

  // Get model information
  async getModelInfo(): Promise<ModelInfo> {
    try {
      const response = await fetch(`${this.baseURL}/model-info`);
      return await response.json();
    } catch (err) {
      console.error('Failed to get model info:', err);
      throw new Error('Failed to get model information');
    }
  }

  // Analyze video with file upload
  async analyzeVideo(request: VideoAnalysisRequest): Promise<AnalysisResponse> {
    try {
      const formData = new FormData();
      formData.append('prompt', request.prompt);
      
      if (request.systemPrompt) {
        formData.append('system_prompt', request.systemPrompt);
      }
      
      if (request.videoFile) {
        formData.append('video_file', request.videoFile);
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
    } catch (error) {
      console.error('Video analysis failed:', error);
      throw new Error('Failed to analyze video');
    }
  }

  // Analyze text only (without video)
  async analyzeTextOnly(prompt: string): Promise<AnalysisResponse> {
    try {
      const response = await fetch(`${this.baseURL}/analyze-text`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          prompt,
          max_tokens: 512,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Text analysis failed');
      }

      return await response.json();
    } catch (error) {
      console.error('Text analysis failed:', error);
      throw new Error('Failed to analyze text');
    }
  }

  // Check if backend is ready
  async isReady(): Promise<boolean> {
    try {
      const health = await this.getHealthStatus();
      return health.status === 'healthy' && health.model_loaded;
    } catch (error) {
      return false;
    }
  }
}

// Export singleton instance
export const cosmosAPI = new CosmosAPIService();

// Export class for custom instances
export default CosmosAPIService;
