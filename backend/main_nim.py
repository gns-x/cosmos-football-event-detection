# Alternative Backend using NVIDIA NIM (No Hugging Face Required)

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import requests
import json
import cv2
import numpy as np
from PIL import Image
import base64
import io
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Cosmos Video Analysis API (NVIDIA NIM)",
    description="Backend API using NVIDIA NIM for Cosmos-Reason1-7B",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables
load_dotenv()

# NVIDIA NIM Configuration
NIM_BASE_URL = os.getenv("NIM_BASE_URL", "https://api.nim.nvidia.com/v1")
NIM_API_KEY = os.getenv("NVIDIA_API_KEY")
NIM_MODEL_NAME = os.getenv("NIM_MODEL_NAME", "cosmos-reason1-7b")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
TIMEOUT_SECONDS = int(os.getenv("TIMEOUT_SECONDS", "60"))

class VideoAnalysisRequest(BaseModel):
    prompt: str
    video_frames: Optional[List[str]] = None
    max_tokens: Optional[int] = 512

class AnalysisResponse(BaseModel):
    reasoning: List[str]
    answer: str
    confidence: float
    timestamp: str
    actor: str

def extract_frames_from_video(video_file) -> List[np.ndarray]:
    """Extract frames from uploaded video file"""
    try:
        temp_path = f"temp_video_{hash(video_file.filename)}.mp4"
        with open(temp_path, "wb") as buffer:
            content = video_file.file.read()
            buffer.write(content)
        
        cap = cv2.VideoCapture(temp_path)
        frames = []
        
        frame_count = 0
        while cap.isOpened() and frame_count < 10:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            frame_count += 1
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + 30)
        
        cap.release()
        os.remove(temp_path)
        
        return frames
        
    except Exception as e:
        logger.error(f"Error extracting frames: {str(e)}")
        return []

def frames_to_base64(frames: List[np.ndarray]) -> List[str]:
    """Convert frames to base64 strings"""
    base64_frames = []
    for frame in frames:
        pil_image = Image.fromarray(frame)
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=85)
        img_str = base64.b64encode(buffer.getvalue()).decode()
        base64_frames.append(img_str)
    
    return base64_frames

def query_nvidia_nim(prompt: str, images: List[str] = None) -> str:
    """Query NVIDIA NIM API for Cosmos-Reason1-7B"""
    try:
        headers = {
            "Authorization": f"Bearer {NIM_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Prepare the request payload
        payload = {
            "model": NIM_MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt}
                    ]
                }
            ],
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE
        }
        
        # Add images if provided
        if images:
            for img_base64 in images[:3]:  # Limit to 3 images
                payload["messages"][0]["content"].append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
                })
        
        # Make API call to NVIDIA NIM
        response = requests.post(
            f"{NIM_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=TIMEOUT_SECONDS
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            logger.error(f"NVIDIA NIM API error: {response.status_code} - {response.text}")
            raise HTTPException(status_code=500, detail="NVIDIA NIM API error")
            
    except Exception as e:
        logger.error(f"Error querying NVIDIA NIM: {str(e)}")
        raise HTTPException(status_code=500, detail=f"NIM query failed: {str(e)}")

def analyze_video_with_nim(prompt: str, frames: List[str] = None) -> AnalysisResponse:
    """Analyze video using NVIDIA NIM Cosmos-Reason1-7B"""
    try:
        # Prepare analysis prompt
        analysis_prompt = f"""
        You are analyzing a football video. Please provide detailed reasoning about the following question:
        
        Question: {prompt}
        
        Please analyze the video content and provide:
        1. Step-by-step reasoning
        2. A clear answer
        3. Confidence level
        4. Key timestamp
        5. Main actor involved
        
        Format your response as structured reasoning followed by a clear answer.
        """
        
        # Query NVIDIA NIM
        response = query_nvidia_nim(analysis_prompt, frames)
        
        # Parse response (simplified)
        lines = response.split('\n')
        reasoning = []
        answer = ""
        
        for line in lines:
            if line.strip().startswith(('1.', '2.', '3.', '4.', '5.', '-', '•')):
                reasoning.append(line.strip())
            elif 'answer' in line.lower() or 'conclusion' in line.lower():
                answer = line.strip()
        
        if not answer:
            answer = lines[-1] if lines else "Analysis completed"
        
        return AnalysisResponse(
            reasoning=reasoning[:5] if reasoning else ["Analysis completed"],
            answer=answer,
            confidence=0.85,
            timestamp="0:00",
            actor="Player"
        )
        
    except Exception as e:
        logger.error(f"Error in video analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Cosmos Video Analysis API (NVIDIA NIM) is running",
        "nim_configured": NIM_API_KEY is not None
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "nim_configured": NIM_API_KEY is not None,
        "model_name": "cosmos-reason1-7b",
        "provider": "NVIDIA NIM"
    }

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_video(
    prompt: str,
    video_file: Optional[UploadFile] = File(None)
):
    """Analyze video using NVIDIA NIM Cosmos-Reason1-7B"""
    
    if not NIM_API_KEY:
        raise HTTPException(status_code=503, detail="NVIDIA API key not configured")
    
    try:
        frames = []
        
        if video_file:
            video_frames = extract_frames_from_video(video_file)
            frames = frames_to_base64(video_frames)
            logger.info(f"Extracted {len(frames)} frames from video")
        
        result = analyze_video_with_nim(prompt, frames)
        return result
        
    except Exception as e:
        logger.error(f"Error in analyze endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-text", response_model=AnalysisResponse)
async def analyze_text_only(request: VideoAnalysisRequest):
    """Analyze text prompt only"""
    
    if not NIM_API_KEY:
        raise HTTPException(status_code=503, detail="NVIDIA API key not configured")
    
    try:
        result = analyze_video_with_nim(request.prompt, request.video_frames)
        return result
        
    except Exception as e:
        logger.error(f"Error in text analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info")
async def get_model_info():
    """Get information about the model"""
    return {
        "model_name": "cosmos-reason1-7b",
        "provider": "NVIDIA NIM",
        "configured": NIM_API_KEY is not None,
        "description": "NVIDIA Cosmos-Reason1-7B via NIM API"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
