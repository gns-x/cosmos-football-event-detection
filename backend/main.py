from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import cv2
import numpy as np
from PIL import Image
import base64
import io
import os
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Cosmos Video Analysis API",
    description="Backend API for video analysis using Cosmos-Reason1-7B model",
    version="1.0.0"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model
model = None
tokenizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"

# Pydantic models
class VideoAnalysisRequest(BaseModel):
    prompt: str
    video_frames: Optional[List[str]] = None  # Base64 encoded frames
    max_tokens: Optional[int] = 512

class AnalysisResponse(BaseModel):
    reasoning: List[str]
    answer: str
    confidence: float
    timestamp: str
    actor: str

@app.on_event("startup")
async def load_model():
    """Load the Cosmos-Reason1-7B model on startup"""
    global model, tokenizer
    
    try:
        logger.info("Loading Cosmos-Reason1-7B model...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("nvidia/Cosmos-Reason1-7B")
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            "nvidia/Cosmos-Reason1-7B",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True
        )
        
        if device == "cpu":
            model = model.to(device)
            
        logger.info(f"Model loaded successfully on {device}")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to load model")

def extract_frames_from_video(video_file) -> List[np.ndarray]:
    """Extract frames from uploaded video file"""
    try:
        # Save uploaded file temporarily
        temp_path = f"temp_video_{hash(video_file.filename)}.mp4"
        with open(temp_path, "wb") as buffer:
            content = video_file.file.read()
            buffer.write(content)
        
        # Extract frames using OpenCV
        cap = cv2.VideoCapture(temp_path)
        frames = []
        
        frame_count = 0
        while cap.isOpened() and frame_count < 10:  # Limit to 10 frames
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            frame_count += 1
            
            # Skip frames to get representative samples
            cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + 30)
        
        cap.release()
        
        # Clean up temp file
        os.remove(temp_path)
        
        return frames
        
    except Exception as e:
        logger.error(f"Error extracting frames: {str(e)}")
        return []

def frames_to_base64(frames: List[np.ndarray]) -> List[str]:
    """Convert frames to base64 strings"""
    base64_frames = []
    for frame in frames:
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(frame)
        
        # Convert to base64
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=85)
        img_str = base64.b64encode(buffer.getvalue()).decode()
        base64_frames.append(img_str)
    
    return base64_frames

def analyze_video_with_cosmos(prompt: str, frames: List[str] = None) -> AnalysisResponse:
    """Analyze video using Cosmos-Reason1-7B model"""
    try:
        # Prepare the input text
        if frames:
            # For now, we'll use text-only analysis
            # In a full implementation, you'd process the frames
            input_text = f"""
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
        else:
            input_text = f"""
            Analyze this football scenario: {prompt}
            
            Provide detailed reasoning and a clear answer about what happened.
            """
        
        # Tokenize input
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract reasoning and answer (simplified parsing)
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
            confidence=0.85,  # Placeholder confidence
            timestamp="0:00",  # Placeholder timestamp
            actor="Player"  # Placeholder actor
        )
        
    except Exception as e:
        logger.error(f"Error in video analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Cosmos Video Analysis API is running", "model_loaded": model is not None}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": device,
        "model_name": "nvidia/Cosmos-Reason1-7B"
    }

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_video(
    prompt: str,
    video_file: Optional[UploadFile] = File(None)
):
    """Analyze video using Cosmos-Reason1-7B model"""
    
    if not model or not tokenizer:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        frames = []
        
        if video_file:
            # Extract frames from video
            video_frames = extract_frames_from_video(video_file)
            frames = frames_to_base64(video_frames)
            logger.info(f"Extracted {len(frames)} frames from video")
        
        # Analyze with Cosmos model
        result = analyze_video_with_cosmos(prompt, frames)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in analyze endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-text", response_model=AnalysisResponse)
async def analyze_text_only(request: VideoAnalysisRequest):
    """Analyze text prompt only (without video)"""
    
    if not model or not tokenizer:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        result = analyze_video_with_cosmos(request.prompt, request.video_frames)
        return result
        
    except Exception as e:
        logger.error(f"Error in text analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model"""
    return {
        "model_name": "nvidia/Cosmos-Reason1-7B",
        "device": device,
        "loaded": model is not None,
        "description": "NVIDIA Cosmos-Reason1-7B for physical AI and embodied reasoning"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
