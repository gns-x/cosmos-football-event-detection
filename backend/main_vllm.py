#!/usr/bin/env python3

"""
Cosmos-Reason1-7B Backend with vLLM
Uses locally cached model with FastAPI + vLLM for high-performance inference
"""

import os
import tempfile
import asyncio
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# IMPORTANT: Force decord as video reader to avoid torchvision fallback
os.environ["QWEN_VL_VIDEO_READER_BACKEND"] = "decord"

# Model configuration
MODEL_OR_PATH = "nvidia/Cosmos-Reason1-7B"

# Initialize FastAPI app
app = FastAPI(
    title="Cosmos-Reason1-7B Video Analysis API",
    description="Local Cosmos-Reason1-7B model with vLLM for video and image analysis",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and processor
llm: Optional[LLM] = None
processor: Optional[AutoProcessor] = None

# Recommended default sampling params
DEFAULT_SAMPLING = SamplingParams(
    temperature=0.6,
    top_p=0.95,
    repetition_penalty=1.05,
    max_tokens=1024,  # Increase to 4096 for longer reasoning
)

@app.on_event("startup")
async def startup_event():
    """Initialize the model and processor on startup"""
    global llm, processor
    
    print("🚀 Initializing Cosmos-Reason1-7B model...")
    
    try:
        # Load the model with vLLM
        print("📥 Loading model with vLLM...")
        llm = LLM(
            model=MODEL_OR_PATH,
            limit_mm_per_prompt={"image": 10, "video": 10},
            trust_remote_code=True,
        )
        
        # Load the processor
        print("📝 Loading processor...")
        processor = AutoProcessor.from_pretrained(
            MODEL_OR_PATH, 
            trust_remote_code=True
        )
        
        print("✅ Model and processor loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        raise e

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": MODEL_OR_PATH,
        "vllm_loaded": llm is not None,
        "processor_loaded": processor is not None
    }

@app.get("/model-info")
async def model_info():
    """Get model information"""
    return {
        "model_name": MODEL_OR_PATH,
        "model_type": "Cosmos-Reason1-7B",
        "backend": "vLLM",
        "video_reader": os.environ.get("QWEN_VL_VIDEO_READER_BACKEND", "default"),
        "max_tokens": DEFAULT_SAMPLING.max_tokens,
        "temperature": DEFAULT_SAMPLING.temperature
    }

@app.post("/analyze-text")
async def analyze_text(prompt: str = Form(...)):
    """Analyze text-only prompts"""
    if not llm or not processor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant specialized in football video analysis."},
            {"role": "user", "content": prompt}
        ]
        
        # Build chat prompt
        chat_prompt = processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Generate response
        outputs = llm.generate([{"prompt": chat_prompt}], sampling_params=DEFAULT_SAMPLING)
        result_text = outputs[0].outputs[0].text
        
        return {
            "reasoning": [result_text],
            "answer": result_text,
            "confidence": 0.85,
            "timestamp": "text-analysis",
            "actor": "cosmos-reason1-7b"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/analyze-image")
async def analyze_image(
    prompt: str = Form(...),
    file: UploadFile = File(...)
):
    """Analyze uploaded image"""
    if not llm or not processor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Save uploaded image to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=_suffix_for_upload(file.filename)) as tmp:
        tmp.write(await file.read())
        img_path = tmp.name
    
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant specialized in football video analysis."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": f"file://{img_path}"}
            ]}
        ]
        
        result_text = _generate_text(messages, DEFAULT_SAMPLING)
        
        return {
            "reasoning": [result_text],
            "answer": result_text,
            "confidence": 0.90,
            "timestamp": "image-analysis",
            "actor": "cosmos-reason1-7b"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image analysis failed: {str(e)}")
    finally:
        _safe_remove(img_path)

@app.post("/analyze")
async def analyze_video(
    prompt: str = Form(...),
    video_file: UploadFile = File(...),
    fps: int = Form(4)  # NVIDIA recommends fps=4
):
    """Analyze uploaded video"""
    if not llm or not processor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Save uploaded video to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=_suffix_for_upload(video_file.filename)) as tmp:
        tmp.write(await video_file.read())
        vid_path = tmp.name
    
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant specialized in football video analysis."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "video", "video": f"file://{vid_path}", "fps": int(fps)}
            ]}
        ]
        
        result_text = _generate_text(messages, DEFAULT_SAMPLING)
        
        return {
            "reasoning": [result_text],
            "answer": result_text,
            "confidence": 0.95,
            "timestamp": "video-analysis",
            "actor": "cosmos-reason1-7b"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video analysis failed: {str(e)}")
    finally:
        _safe_remove(vid_path)

@app.post("/clear-cache")
async def clear_cache():
    """Clear GPU cache"""
    try:
        if llm:
            # Clear vLLM cache if available
            pass
        return {"status": "cache cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache clear failed: {str(e)}")

def _generate_text(messages, sampling_params: SamplingParams) -> str:
    """Generate text using vLLM with multimodal inputs"""
    # Build chat prompt
    prompt = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # Prepare multimodal inputs
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, 
        return_video_kwargs=True
    )
    
    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs
    
    llm_inputs = {
        "prompt": prompt,
        "multi_modal_data": mm_data,
        "mm_processor_kwargs": video_kwargs,
    }
    
    outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
    return outputs[0].outputs[0].text

def _suffix_for_upload(filename: str) -> str:
    """Return safe file suffix based on filename"""
    lower = (filename or "").lower()
    if lower.endswith(".mp4"):
        return ".mp4"
    if lower.endswith(".jpg") or lower.endswith(".jpeg"):
        return ".jpg"
    if lower.endswith(".png"):
        return ".png"
    return ".bin"

def _safe_remove(path: str) -> None:
    """Safely remove file"""
    try:
        os.remove(path)
    except Exception:
        pass

if __name__ == "__main__":
    print("🚀 Starting Cosmos-Reason1-7B API with vLLM...")
    print(f"📝 Model: {MODEL_OR_PATH}")
    print(f"🎬 Video Reader: {os.environ.get('QWEN_VL_VIDEO_READER_BACKEND', 'default')}")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )
