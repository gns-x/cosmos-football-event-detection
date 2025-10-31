import os
import sys
import json
import re
import subprocess
import warnings
import random
import gc
from typing import Optional, List, Dict, Any
from pathlib import Path
import threading
import time

warnings.simplefilter("ignore")

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

try:
    import torch
    from transformers import pipeline, BitsAndBytesConfig, AutoConfig, AutoProcessor, AutoModelForImageTextToText
    from peft import PeftModel
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch/Transformers not available. Model operations will be disabled.")

# Paths (relative to train/ root)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(ROOT_DIR, "cosmos-model")
ADAPTER_DIR = os.path.join(ROOT_DIR, "04_model_output", "final_checkpoint")
CLIPS_DIR = os.path.join(ROOT_DIR, "01_clips")
ANNO_DIR = os.path.join(ROOT_DIR, "02_annotations")
DATASET_DIR = os.path.join(ROOT_DIR, "03_dataset")
OUTPUT_DIR = os.path.join(ROOT_DIR, "04_model_output")
# Temp folder for uploaded videos
TEMP_DIR = os.path.join(os.path.dirname(__file__), "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

# Event classes
CLASSES = [
    "penalty_shot", "goal", "goal_line_event", "woodworks",
    "shot_on_target", "red_card", "yellow_card", "hat_trick"
]

# Training status tracking
_training_status = {"running": False, "progress": 0, "current_step": "", "logs": []}
_training_lock = threading.Lock()

app = FastAPI()

# Allow vite dev server default origin and any localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HealthResponse(BaseModel):
    status: str
    nim_ready: bool
    model: str


class AnalyzeResponse(BaseModel):
    reasoning: list[str]
    answer: str
    confidence: float
    timestamp: str
    actor: str
    events: Optional[list[dict]] = None
    summary: Optional[dict] = None


_pipe = None           # trained (with adapters if present)
_processor = None
_base_pipe = None      # base-only (no adapters)


def check_model_exists():
    """Check if model directory exists"""
    return os.path.isdir(MODEL_PATH)

def get_pipe():
    global _pipe
    global _processor
    
    if not TORCH_AVAILABLE:
        raise HTTPException(status_code=503, detail="PyTorch/Transformers not installed")
    
    if not check_model_exists():
        raise HTTPException(status_code=404, detail=f"Model not found at {MODEL_PATH}. Please ensure the cosmos-model directory exists.")
    
    if _pipe is not None:
        return _pipe
    print("--- Starting inference test... ---")
    print("Loading pipeline for task: 'image-text-to-text'")
    print("This will load the model, processor, and config... This may take a minute...")
    
    try:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        # Load processor from adapter checkpoint if present (includes any special tokens)
        proc_dir = ADAPTER_DIR if os.path.isdir(ADAPTER_DIR) else MODEL_PATH
        _processor = AutoProcessor.from_pretrained(proc_dir, trust_remote_code=True)
        if getattr(_processor, "tokenizer", None) and _processor.tokenizer.pad_token is None:
            _processor.tokenizer.pad_token = _processor.tokenizer.eos_token

        # Load base model in 4-bit then attach LoRA adapters if available
        config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
        base_model = AutoModelForImageTextToText.from_pretrained(
            MODEL_PATH,
            config=config,
            quantization_config=quant_config,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        if os.path.isdir(ADAPTER_DIR):
            print("Merging LoRA adapters into base model...")
            peft_wrapped = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
            # Prefer a fully merged model to keep original class for the pipeline task
            try:
                model = peft_wrapped.merge_and_unload()
            except Exception:
                # Fallback: use the PEFT-wrapped model directly
                model = peft_wrapped
        else:
            model = base_model

        _pipe = pipeline(
            "image-text-to-text",
            model=model,
            processor=_processor,
            device_map="auto",
            trust_remote_code=True,
        )
        print("--- Pipeline created successfully. ---")
        return _pipe
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


def get_base_pipe():
    """Load base model ONLY (no adapters) for Analyze page."""
    global _base_pipe
    global _processor
    if not TORCH_AVAILABLE:
        raise HTTPException(status_code=503, detail="PyTorch/Transformers not installed")
    if not check_model_exists():
        raise HTTPException(status_code=404, detail=f"Model not found at {MODEL_PATH}.")
    if _base_pipe is not None:
        return _base_pipe
    try:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        # Always load processor from base model for base-only pipeline
        _processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
        if getattr(_processor, "tokenizer", None) and _processor.tokenizer.pad_token is None:
            _processor.tokenizer.pad_token = _processor.tokenizer.eos_token

        config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_PATH,
            config=config,
            quantization_config=quant_config,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        _base_pipe = pipeline(
            "image-text-to-text",
            model=model,
            processor=_processor,
            device_map="auto",
            trust_remote_code=True,
        )
        return _base_pipe
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load base model: {str(e)}")


@app.get("/health", response_model=HealthResponse)
def health():
    try:
        p = get_pipe()
        ready = p is not None
        return HealthResponse(status="healthy" if ready else "loading", nim_ready=ready, model=os.path.basename(MODEL_PATH))
    except Exception as e:
        print(f"Health check error: {e}")
        return HealthResponse(status="offline", nim_ready=False, model=os.path.basename(MODEL_PATH))


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    prompt: str = Form(...),
    system_prompt: Optional[str] = Form(None),
    file: UploadFile = File(...),
):
    # Save uploaded file to temp
    dst_path = os.path.join(TEMP_DIR, file.filename)
    with open(dst_path, "wb") as f:
        f.write(await file.read())

    # Compose chat messages following evaluate.py style
    # Put <video> tag in text and attach the video separately
    content = []
    if system_prompt:
        content.append({"type": "text", "text": system_prompt})
    content.append({"type": "text", "text": f"<video>\n{prompt}"})
    content.append({"type": "video", "video": dst_path})
    messages = [{"role": "user", "content": content}]

    # Use BASE model only for Analyze page
    p = get_base_pipe()
    output = p(messages, max_new_tokens=512, temperature=0.3, top_p=0.9)
    text = ""
    try:
        text = output[0]["generated_text"][-1]["content"]
    except Exception:
        text = str(output)

    # Try to extract events JSON if the model emitted structured output
    events = []
    try:
        import re, json as _json
        # Prefer content inside <json>...</json> if present
        m_json = re.search(r"<json>([\s\S]*?)</json>", text, re.I)
        candidate = m_json.group(1) if m_json else text
        # Strip markdown fences if present
        candidate = re.sub(r"^```[a-zA-Z]*\n", "", candidate.strip())
        candidate = re.sub(r"\n```$", "", candidate)
        m_block = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", candidate, re.S)
        payload = m_block.group(1) if m_block else candidate
        data = _json.loads(payload)
        if isinstance(data, dict) and "all_events" in data and isinstance(data["all_events"], list):
            events = data["all_events"]
        elif isinstance(data, list):
            events = data
    except Exception:
        events = []

    return AnalyzeResponse(
        reasoning=[],
        answer=text,
        confidence=0.0,
        timestamp="",
        actor=(os.path.basename(MODEL_PATH) + ("+lora" if os.path.isdir(ADAPTER_DIR) else "")),
        events=events,
        summary={},
    )


# ========== PIPELINE ENDPOINTS ==========

class PipelineStatusResponse(BaseModel):
    has_clips: bool
    has_annotations: bool
    has_dataset: bool
    has_model: bool
    has_trained: bool
    clips_count: int
    annotations_count: int
    dataset_count: int

class GenerateAnnotationsResponse(BaseModel):
    success: bool
    message: str
    processed: int
    generated: int

class CreateDatasetResponse(BaseModel):
    success: bool
    message: str
    records: int

class TrainResponse(BaseModel):
    success: bool
    message: str
    job_id: str

class TestInferenceResponse(BaseModel):
    success: bool
    events: List[Dict[str, Any]]
    raw_output: str
    error: Optional[str] = None


def get_duration_seconds(path: str) -> float | None:
    try:
        out = subprocess.check_output([
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', path
        ], stderr=subprocess.DEVNULL).decode().strip()
        return float(out)
    except Exception:
        return None

def to_mmss(val) -> str:
    try:
        sec = float(val)
    except Exception:
        s = str(val)
        if re.match(r'^\d{1,2}:\d{2}$', s):
            return s
        mnum = re.search(r'(\d+(?:\.\d+)?)', s)
        sec = float(mnum.group(1)) if mnum else 0.0
    if sec < 0:
        sec = 0.0
    total = int(round(sec))
    mm = total // 60
    ss = total % 60
    return f"{mm:02d}:{ss:02d}"

def to_seconds(val) -> float:
    try:
        return float(val)
    except Exception:
        s = str(val)
        m = re.match(r'^(\d{1,2}):(\d{2})$', s)
        if m:
            return int(m.group(1)) * 60 + int(m.group(2))
        mnum = re.search(r'(\d+(?:\.\d+)?)', s)
        return float(mnum.group(1)) if mnum else 0.0


@app.get("/pipeline/status", response_model=PipelineStatusResponse)
def get_pipeline_status():
    """Get status of each pipeline step"""
    clips_count = 0
    annotations_count = 0
    dataset_count = 0
    
    if os.path.isdir(CLIPS_DIR):
        for class_name in CLASSES:
            class_clips_dir = os.path.join(CLIPS_DIR, class_name)
            if os.path.isdir(class_clips_dir):
                clips_count += len([f for f in os.listdir(class_clips_dir) if f.lower().endswith('.mp4')])
    
    if os.path.isdir(ANNO_DIR):
        for class_name in CLASSES:
            class_anno_dir = os.path.join(ANNO_DIR, class_name)
            if os.path.isdir(class_anno_dir):
                annotations_count += len([f for f in os.listdir(class_anno_dir) if f.lower().endswith('.json')])
    
    dataset_file = os.path.join(DATASET_DIR, "train_dataset.jsonl")
    if os.path.isfile(dataset_file):
        with open(dataset_file, 'r', encoding='utf-8') as f:
            dataset_count = sum(1 for _ in f)
    
    return PipelineStatusResponse(
        has_clips=clips_count > 0,
        has_annotations=annotations_count > 0,
        has_dataset=dataset_count > 0,
        has_model=check_model_exists(),
        has_trained=os.path.isdir(ADAPTER_DIR),
        clips_count=clips_count,
        annotations_count=annotations_count,
        dataset_count=dataset_count,
    )


@app.post("/pipeline/generate-annotations", response_model=GenerateAnnotationsResponse)
async def generate_annotations():
    """Generate annotations for all clips using the cosmos model"""
    if not TORCH_AVAILABLE:
        return GenerateAnnotationsResponse(
            success=False,
            message="PyTorch/Transformers not installed",
            processed=0,
            generated=0,
        )
    
    if not check_model_exists():
        return GenerateAnnotationsResponse(
            success=False,
            message=f"Model not found at {MODEL_PATH}",
            processed=0,
            generated=0,
        )
    
    if not os.path.isdir(CLIPS_DIR):
        return GenerateAnnotationsResponse(
            success=False,
            message=f"Clips directory not found: {CLIPS_DIR}",
            processed=0,
            generated=0,
        )
    
    # Load pipeline (reuse existing or create new)
    try:
        pipe = get_pipe()
    except Exception as e:
        return GenerateAnnotationsResponse(
            success=False,
            message=f"Failed to load model: {str(e)}",
            processed=0,
            generated=0,
        )
    
    # System prompt from generate_annotations.py
    SYSTEM_PROMPT = (
        "You are a soccer event detection system analyzing broadcast videos. Detect only these events using visual evidence:\n\n"
        "Events: Penalty Shot, Goal, Goal-Line Event, Woodworks, Shot on Target, Red Card, Yellow Card, Hat-Trick.\n\n"
        "Output Format: JSON array with event, start_time (mm:ss), end_time (mm:ss), description, explanation.\n\n"
        "Strictness: Output ONLY the JSON array. No extra text."
    )
    USER_PROMPT = "<video>\nAnalyze this clip and produce only the specified JSON array with events, using mm:ss timestamps relative to the clip (starting from 00:00)."
    
    total_processed = 0
    total_generated = 0
    
    for class_name in CLASSES:
        class_clips_dir = os.path.join(CLIPS_DIR, class_name)
        class_anno_dir = os.path.join(ANNO_DIR, class_name)
        
        if not os.path.isdir(class_clips_dir):
            continue
        
        os.makedirs(class_anno_dir, exist_ok=True)
        clips = sorted([f for f in os.listdir(class_clips_dir) if f.lower().endswith('.mp4')])
        
        for clip_file in clips:
            clip_path = os.path.join(class_clips_dir, clip_file)
            clip_name = os.path.splitext(clip_file)[0]
            anno_path = os.path.join(class_anno_dir, f"{clip_name}.json")
            
            if os.path.isfile(anno_path):
                continue
            
            try:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": SYSTEM_PROMPT},
                            {"type": "text", "text": USER_PROMPT},
                            {"type": "video", "video": clip_path}
                        ]
                    }
                ]
                
                with torch.inference_mode():
                    output = pipe(messages, max_new_tokens=384, temperature=0.4, top_p=0.9)
                
                text = output[0]["generated_text"][-1]["content"]
                
                # Extract JSON
                events = []
                try:
                    m_ans = re.search(r'<answer>([\s\S]*?)</answer>', text, re.I)
                    candidate = m_ans.group(1) if m_ans else text
                    m = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', candidate, re.S)
                    payload = m.group(1) if m else candidate
                    data = json.loads(payload)
                    
                    if isinstance(data, dict) and 'all_events' in data:
                        events = data['all_events']
                    elif isinstance(data, list):
                        events = data
                    
                    # Normalize timestamps
                    clip_duration = get_duration_seconds(clip_path)
                    for ev in events:
                        if 'start_time' in ev:
                            ev['start_time'] = to_mmss(to_seconds(ev['start_time']))
                        if 'end_time' in ev:
                            ev['end_time'] = to_mmss(to_seconds(ev['end_time']))
                except Exception:
                    events = []
                
                with open(anno_path, 'w', encoding='utf-8') as f:
                    json.dump(events, f, ensure_ascii=False, indent=2)
                
                if events:
                    total_generated += 1
                total_processed += 1
            except Exception as e:
                print(f"Error processing {clip_file}: {e}")
    
    return GenerateAnnotationsResponse(
        success=True,
        message=f"Processed {total_processed} clips, generated {total_generated} annotations",
        processed=total_processed,
        generated=total_generated,
    )


@app.post("/pipeline/create-dataset", response_model=CreateDatasetResponse)
async def create_dataset():
    """Create training dataset from annotations"""
    if not os.path.isdir(ANNO_DIR):
        return CreateDatasetResponse(
            success=False,
            message=f"Annotations directory not found: {ANNO_DIR}",
            records=0,
        )
    
    os.makedirs(DATASET_DIR, exist_ok=True)
    dataset_file = os.path.join(DATASET_DIR, "train_dataset.jsonl")
    
    HUMAN_PROMPT = "Analyze the events in this football clip and provide the output in JSON format."
    pairs = []
    
    for event_class in sorted(os.listdir(ANNO_DIR)):
        anno_class_dir = os.path.join(ANNO_DIR, event_class)
        clips_class_dir = os.path.join(CLIPS_DIR, event_class)
        if not os.path.isdir(anno_class_dir) or not os.path.isdir(clips_class_dir):
            continue
        
        for fname in sorted(os.listdir(anno_class_dir)):
            if not fname.lower().endswith(".json"):
                continue
            
            base = os.path.splitext(fname)[0]
            json_abs = os.path.join(anno_class_dir, fname)
            mp4_abs = os.path.join(clips_class_dir, f"{base}.mp4")
            
            if os.path.isfile(json_abs) and os.path.isfile(mp4_abs):
                pairs.append((event_class, base, json_abs, mp4_abs))
    
    if not pairs:
        return CreateDatasetResponse(
            success=False,
            message="No annotation/clip pairs found",
            records=0,
        )
    
    random.shuffle(pairs)
    written = 0
    
    with open(dataset_file, 'w', encoding='utf-8') as out:
        for event_class, base, json_abs, mp4_abs in pairs:
            try:
                with open(json_abs, 'r', encoding='utf-8') as f:
                    events = json.load(f)
                
                if not events:
                    continue
                
                video_abs = os.path.abspath(mp4_abs)
                gpt_response_string = f"```json\n{json.dumps(events, indent=2, ensure_ascii=False)}\n```"
                
                record = {
                    "id": base,
                    "video": [video_abs],
                    "conversations": [
                        {"from": "human", "value": f"<video>\n{HUMAN_PROMPT}"},
                        {"from": "gpt", "value": gpt_response_string}
                    ]
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
            except Exception as e:
                print(f"Error processing {base}: {e}")
    
    return CreateDatasetResponse(
        success=True,
        message=f"Created dataset with {written} records",
        records=written,
    )


@app.post("/pipeline/train", response_model=TrainResponse)
async def train_model(background_tasks: BackgroundTasks):
    """Start training process"""
    if not TORCH_AVAILABLE:
        return TrainResponse(
            success=False,
            message="PyTorch/Transformers not installed",
            job_id="",
        )
    
    if not check_model_exists():
        return TrainResponse(
            success=False,
            message=f"Model not found at {MODEL_PATH}",
            job_id="",
        )
    
    dataset_file = os.path.join(DATASET_DIR, "train_dataset.jsonl")
    if not os.path.isfile(dataset_file):
        return TrainResponse(
            success=False,
            message=f"Dataset not found: {dataset_file}",
            job_id="",
        )
    
    with _training_lock:
        if _training_status["running"]:
            return TrainResponse(
                success=False,
                message="Training already in progress",
                job_id="train_1",
            )
        _training_status["running"] = True
        _training_status["progress"] = 0
        _training_status["current_step"] = "Initializing..."
        _training_status["logs"] = []
    
    job_id = f"train_{int(time.time())}"
    
    # Note: Actual training would run in background
    # For now, return success with message that training would start
    return TrainResponse(
        success=True,
        message="Training started (run training script manually: python 05_scripts/train.py)",
        job_id=job_id,
    )


@app.get("/pipeline/train/status")
def get_training_status():
    """Get training status"""
    with _training_lock:
        return _training_status


@app.post("/pipeline/test-inference", response_model=TestInferenceResponse)
async def test_inference(
    file: UploadFile = File(...),
    prompt: Optional[str] = Form(None),
    system_prompt: Optional[str] = Form(None),
):
    """Test inference on uploaded video using trained model by running test_inference.py script"""
    if not check_model_exists():
        return TestInferenceResponse(
            success=False,
            events=[],
            raw_output="",
            error=f"Model not found at {MODEL_PATH}",
        )
    
    if not os.path.isdir(ADAPTER_DIR):
        return TestInferenceResponse(
            success=False,
            events=[],
            raw_output="",
            error="Trained model not found. Please train the model first.",
        )
    
    # Save uploaded file
    dst_path = os.path.join(TEMP_DIR, file.filename)
    with open(dst_path, "wb") as f:
        f.write(await file.read())
    
    # Run test_inference.py script
    script_path = os.path.join(ROOT_DIR, "05_scripts", "test_inference.py")
    if not os.path.isfile(script_path):
        return TestInferenceResponse(
            success=False,
            events=[],
            raw_output="",
            error=f"Script not found: {script_path}",
        )
    
    try:
        # Run the script and capture output
        cmd = [sys.executable, script_path, dst_path]
        if prompt and prompt.strip():
            cmd.extend(["--prompt", prompt])
        else:
            # Require prompt (no hardcoded default)
            return TestInferenceResponse(
                success=False,
                events=[],
                raw_output="",
                error="Missing required 'prompt' for test inference",
            )
        if system_prompt and system_prompt.strip():
            cmd.extend(["--system-prompt", system_prompt])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=ROOT_DIR,
            timeout=600,  # 10 minute timeout
        )
        
        # Parse stdout for JSON output
        # The script prints normalized JSON (mm:ss format) before the end
        stdout = result.stdout
        stderr = result.stderr
        
        # Extract JSON from output
        # The script prints "--- Normalized JSON (mm:ss) ---" followed by JSON
        events = []
        raw_output = stdout
        
        # Try to find the JSON block after "--- Normalized JSON (mm:ss) ---"
        json_match = re.search(r'--- Normalized JSON \(mm:ss\) ---\s*\n(\{[\s\S]*?\n\})', stdout, re.MULTILINE | re.DOTALL)
        if json_match:
            try:
                json_text = json_match.group(1).strip()
                json_data = json.loads(json_text)
                if isinstance(json_data, dict) and "events" in json_data:
                    events = json_data["events"]
            except Exception as parse_err:
                # If parsing fails, try to extract balanced JSON manually
                try:
                    # Find the last complete JSON object containing "events"
                    json_pattern = r'(\{[\s\S]*?"events"[\s\S]*?\})'
                    matches = list(re.finditer(json_pattern, stdout, re.MULTILINE | re.DOTALL))
                    if matches:
                        json_data = json.loads(matches[-1].group(1))
                        if isinstance(json_data, dict) and "events" in json_data:
                            events = json_data["events"]
                except Exception:
                    pass
        
        # Fallback: try to find any JSON structure with "events" key
        if not events:
            # Look for the last complete JSON structure containing "events"
            json_pattern = r'(\{[\s\S]*?"events"[\s\S]*?\})'
            matches = list(re.finditer(json_pattern, stdout, re.MULTILINE | re.DOTALL))
            if matches:
                for match in reversed(matches):  # Try from last to first
                    try:
                        json_data = json.loads(match.group(1))
                        if isinstance(json_data, dict) and "events" in json_data and isinstance(json_data["events"], list):
                            events = json_data["events"]
                            break
                    except Exception:
                        continue
        
        if result.returncode != 0:
            error_msg = f"Script failed with return code {result.returncode}"
            if stderr:
                error_msg += f": {stderr}"
            return TestInferenceResponse(
                success=False,
                events=events,
                raw_output=stdout,
                error=error_msg,
            )
        
        return TestInferenceResponse(
            success=True,
            events=events if isinstance(events, list) else [],
            raw_output=stdout,
            error=None,
        )
    except subprocess.TimeoutExpired:
        return TestInferenceResponse(
            success=False,
            events=[],
            raw_output="",
            error="Script execution timed out (exceeded 10 minutes)",
        )
    except Exception as e:
        return TestInferenceResponse(
            success=False,
            events=[],
            raw_output="",
            error=f"Failed to run test_inference.py: {str(e)}",
        )


# ========== ANNOTATION API ENDPOINTS ==========

@app.get("/api/annotate/classes")
async def get_classes():
    """Get list of classes (directories) in 01_clips"""
    if not os.path.isdir(CLIPS_DIR):
        return []
    classes = []
    for item in os.listdir(CLIPS_DIR):
        class_path = os.path.join(CLIPS_DIR, item)
        if os.path.isdir(class_path):
            classes.append(item)
    return sorted(classes)


@app.get("/api/annotate/clips/{class_name}")
async def get_clips(class_name: str):
    """Get list of clips in a class"""
    class_clips_dir = os.path.join(CLIPS_DIR, class_name)
    if not os.path.isdir(class_clips_dir):
        raise HTTPException(status_code=404, detail=f"Class not found: {class_name}")
    
    clips = []
    for filename in os.listdir(class_clips_dir):
        if filename.lower().endswith('.mp4'):
            name = os.path.splitext(filename)[0]
            anno_path = os.path.join(ANNO_DIR, class_name, f"{name}.json")
            has_annotation = os.path.isfile(anno_path)
            clips.append({"name": name, "hasAnnotation": has_annotation})
    return sorted(clips, key=lambda x: x["name"])


@app.get("/api/annotate/video/{class_name}/{clip_name}")
async def get_video(class_name: str, clip_name: str):
    """Get video file for a clip"""
    video_path = os.path.join(CLIPS_DIR, class_name, f"{clip_name}.mp4")
    if not os.path.isfile(video_path):
        raise HTTPException(status_code=404, detail=f"Video not found: {class_name}/{clip_name}")
    
    from fastapi.responses import FileResponse
    return FileResponse(video_path, media_type="video/mp4")


@app.get("/api/annotate/annotation/{class_name}/{clip_name}")
async def get_annotation(class_name: str, clip_name: str):
    """Get annotation JSON for a clip"""
    anno_path = os.path.join(ANNO_DIR, class_name, f"{clip_name}.json")
    if not os.path.isfile(anno_path):
        return []
    
    try:
        with open(anno_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read annotation: {str(e)}")


@app.post("/api/annotate/annotation/{class_name}/{clip_name}")
async def save_annotation(class_name: str, clip_name: str, events: List[Dict[str, Any]]):
    """Save annotation JSON for a clip"""
    anno_class_dir = os.path.join(ANNO_DIR, class_name)
    os.makedirs(anno_class_dir, exist_ok=True)
    
    anno_path = os.path.join(anno_class_dir, f"{clip_name}.json")
    try:
        with open(anno_path, 'w', encoding='utf-8') as f:
            json.dump(events, f, indent=2, ensure_ascii=False)
        return {"success": True, "message": "Annotation saved"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save annotation: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))


