from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import cv2
import numpy as np
from PIL import Image
import base64
import io
import os
from dotenv import load_dotenv
import logging
import gc

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Cosmos Video Analysis API (Local)",
    description="Local deployment of Cosmos-Reason1-7B using Hugging Face",
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

# Configuration
MODEL_NAME = os.getenv("MODEL_NAME", "nvidia/Cosmos-Reason1-7B")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_QUANTIZATION = os.getenv("USE_QUANTIZATION", "true").lower() == "true"
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))

# Global variables for model
model = None
tokenizer = None

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

def get_quantization_config():
    """Get quantization configuration for GPU memory optimization"""
    if not torch.cuda.is_available():
        return None
    
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

def load_model():
    """Load the Cosmos-Reason1-7B model locally"""
    global model, tokenizer
    
    try:
        logger.info(f"Loading model: {MODEL_NAME}")
        logger.info(f"Device: {DEVICE}")
        logger.info(f"Quantization: {USE_QUANTIZATION}")
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            use_fast=True
        )
        
        # Set pad token if not exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with optimization
        logger.info("Loading model...")
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if DEVICE == "cuda" else torch.float32,
            "device_map": "auto" if DEVICE == "cuda" else None,
            "low_cpu_mem_usage": True,
        }
        
        # Add quantization if enabled and GPU available
        if USE_QUANTIZATION and DEVICE == "cuda":
            quantization_config = get_quantization_config()
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
                logger.info("Using 4-bit quantization for memory optimization")
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            **model_kwargs
        )
        
        if DEVICE == "cpu":
            model = model.to(DEVICE)
        
        logger.info("Model loaded successfully!")
        
        # Log memory usage
        if DEVICE == "cuda":
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"GPU Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
        
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

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

def analyze_video_local(prompt: str, frames: List[str] = None) -> AnalysisResponse:
    """Analyze video using local Cosmos-Reason1-7B model"""
    global model, tokenizer
    
    if not model or not tokenizer:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Prepare the input text
        if frames:
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
        inputs = tokenizer(
            input_text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=1024,
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove input from response
        if input_text in response:
            response = response.replace(input_text, "").strip()
        
        # Extract reasoning and answer (simplified parsing)
        lines = response.split('\n')
        reasoning = []
        answer = ""
        
        for line in lines:
            line = line.strip()
            if line.startswith(('1.', '2.', '3.', '4.', '5.', '-', '•')):
                reasoning.append(line)
            elif 'answer' in line.lower() or 'conclusion' in line.lower():
                answer = line
        
        if not answer:
            answer = lines[-1] if lines else "Analysis completed"
        
        # Clean up memory
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        
        return AnalysisResponse(
            reasoning=reasoning[:5] if reasoning else ["Analysis completed"],
            answer=answer,
            confidence=0.85,
            timestamp="0:00",
            actor="Player"
        )
        
    except Exception as e:
        logger.error(f"Error in video analysis: {str(e)}")
        # Clean up memory on error
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("Starting up Cosmos Video Analysis API...")
    success = load_model()
    if not success:
        logger.error("Failed to load model on startup")
        raise HTTPException(status_code=500, detail="Failed to load model")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Cosmos Video Analysis API (Local) is running",
        "model_loaded": model is not None,
        "device": DEVICE,
        "model_name": MODEL_NAME
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    memory_info = {}
    if DEVICE == "cuda":
        memory_info = {
            "gpu_memory_allocated": f"{torch.cuda.memory_allocated() / 1024**3:.2f}GB",
            "gpu_memory_reserved": f"{torch.cuda.memory_reserved() / 1024**3:.2f}GB"
        }
    
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": DEVICE,
        "model_name": MODEL_NAME,
        "quantization": USE_QUANTIZATION,
        **memory_info
    }

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_video(
    prompt: str,
    video_file: Optional[UploadFile] = File(None)
):
    """Analyze video using local Cosmos-Reason1-7B model"""
    
    if not model or not tokenizer:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        frames = []
        
        if video_file:
            video_frames = extract_frames_from_video(video_file)
            frames = frames_to_base64(video_frames)
            logger.info(f"Extracted {len(frames)} frames from video")
        
        result = analyze_video_local(prompt, frames)
        return result
        
    except Exception as e:
        logger.error(f"Error in analyze endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-text", response_model=AnalysisResponse)
async def analyze_text_only(request: VideoAnalysisRequest):
    """Analyze text prompt only"""
    
    if not model or not tokenizer:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        result = analyze_video_local(request.prompt, request.video_frames)
        return result
        
    except Exception as e:
        logger.error(f"Error in text analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model"""
    memory_info = {}
    if DEVICE == "cuda":
        memory_info = {
            "gpu_memory_allocated": f"{torch.cuda.memory_allocated() / 1024**3:.2f}GB",
            "gpu_memory_reserved": f"{torch.cuda.memory_reserved() / 1024**3:.2f}GB"
        }
    
    return {
        "model_name": MODEL_NAME,
        "device": DEVICE,
        "loaded": model is not None,
        "quantization": USE_QUANTIZATION,
        "description": "NVIDIA Cosmos-Reason1-7B (Local Deployment)",
        **memory_info
    }

@app.post("/clear-cache")
async def clear_cache():
    """Clear GPU memory cache"""
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        gc.collect()
        return {"message": "GPU cache cleared"}
    return {"message": "No GPU cache to clear"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
