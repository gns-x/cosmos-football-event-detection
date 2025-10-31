#!/usr/bin/env python3
"""
Evaluation Script (frontend-driven prompts + JSON repair + timing fixes)
- Loads base model + LoRA adapters (trained model)
- Runs inference on a video clip using prompts passed from the frontend
- Extracts JSON (answer/fenced/balanced)
- Repairs common JSON glitches (e.g., missing comma between fields)
- Normalizes start/end times to numeric seconds and mm:ss strings
"""

import os
import sys
import re
import json
import warnings
import argparse
import math
import torch

from peft import PeftModel
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
    pipeline,
)

warnings.simplefilter("ignore")

# ---------------- Paths ----------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
MODEL_DIR = os.path.join(ROOT_DIR, "cosmos-model")
ADAPTER_DIR = os.path.join(ROOT_DIR, "04_model_output/final_checkpoint")

# ---------------- Prompts ----------------
# No hardcoded prompts; both prompts must be provided via CLI flags

# ---------------- Timing normalization knobs ----------------
PRE_EVENT_SECONDS = 2.0      # window before a detected instant timestamp
POST_EVENT_SECONDS = 6.0     # window after a detected instant timestamp
DEFAULT_EVENT_LEN = 8.0      # default length when no timestamp is found
CLIP_DURATION_SECONDS = None # set if you want to clamp end_time to clip duration

# ---------------- CLI ----------------
def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate video and emit JSON with normalized start/end times.")
    ap.add_argument("video", help="Path to video file.")
    ap.add_argument("--json-out", help="Optional path to save JSON output.", default=None)
    ap.add_argument("--prompt", dest="user_prompt", help="User prompt text from frontend.", required=True)
    ap.add_argument("--system-prompt", dest="system_prompt", help="System prompt text from frontend.", required=False, default=None)
    return ap.parse_args()

# ---------------- Utilities: time parsing/formatting ----------------
TEXTUAL_BAD_VALUES = {
    "period", "first half", "second half", "ht", "half-time", "halftime",
    "extra time", "et", "overtime", "ft", "full time"
}

def to_seconds_from_text(s: str, default=None):
    """
    Accepts:
      - "mm:ss", "m:ss", "mm:ss.s", "ss", "ss.s", "10.5 s"
      - plain float string "12.3"
    Returns float seconds or default.
    """
    if not isinstance(s, str):
        return default
    t = s.strip().lower()
    if t in TEXTUAL_BAD_VALUES:
        return default

    # "10.5 s", "10s", "10 sec"
    m = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*(s|sec|secs|second|seconds)?\s*$", t)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return default

    # "mm:ss" or "mm:ss.s"
    m = re.match(r"^\s*(\d{1,2}):([0-5]?\d(?:\.\d{1,3})?)\s*$", t)
    if m:
        mins = int(m.group(1))
        secs = float(m.group(2))
        return mins * 60.0 + secs

    # plain float
    try:
        return float(t)
    except Exception:
        return default

def coerce_seconds(val, default=None):
    """Coerce various types to float seconds, rejecting textual labels."""
    if isinstance(val, (int, float)) and math.isfinite(val):
        return float(val)
    if isinstance(val, str):
        return to_seconds_from_text(val, default=default)
    return default

def to_mmss(seconds: float) -> str:
    """Format seconds as MM:SS (rounded down)."""
    if seconds is None or not math.isfinite(seconds):
        return "00:00"
    s = max(0.0, seconds)
    m = int(s // 60)
    sec = int(s % 60)
    return f"{m:02d}:{sec:02d}"

# ---------------- JSON extraction ----------------
def extract_fenced_json(text: str) -> str | None:
    m = re.search(r"```json\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else None

def extract_answer_json(text: str) -> str | None:
    m = re.search(r"<answer>\s*(\{.*?\})\s*</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else None

def extract_balanced_json(text: str) -> str | None:
    s = text
    n = len(s)
    best = None
    i = 0
    while i < n:
        if s[i] == "{":
            depth = 0
            j = i
            in_str = False
            esc = False
            while j < n:
                ch = s[j]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                else:
                    if ch == '"':
                        in_str = True
                    elif ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            candidate = s[i:j+1]
                            if best is None or len(candidate) > len(best):
                                best = candidate
                            break
                j += 1
        i += 1
    return best

def try_parse_json(s: str) -> dict | list | None:
    if s is None:
        return None
    try:
        return json.loads(s)
    except Exception:
        return None

# ---------------- JSON sanitizer (auto-repair) ----------------
def fix_common_json_issues(text: str) -> str:
    """
    Attempt to repair common JSON issues produced by the model:
      - Missing comma between adjacent fields, e.g., description ... "explanation"
      - Smart quotes -> regular quotes
      - Wrap top-level array into {"all_events": ...} where appropriate
    """
    if text is None:
        return text

    fixed = text

    # Normalize smart quotes to ASCII quotes
    fixed = fixed.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")

    # Insert missing comma between "description": "..."\s*"explanation":
    # Matches: "description": "...."   "explanation":
    fixed = re.sub(
        r'("description"\s*:\s*"[^"]*")\s*("explanation"\s*:)',
        r'\1,\n\2',
        fixed,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # If the top-level is clearly an array (starts with [ and ends with ]), wrap it
    # This helps when the model returns `[ {...}, {...} ]` instead of the requested object.
    stripped = fixed.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        fixed = '{"all_events": ' + stripped + '}'

    return fixed

# ---------------- Normalization of event list & timings ----------------
def ensure_events_list(model_json) -> list[dict]:
    """
    Accepts:
      - {"all_events": [ ... ]}
      - {"events": [ ... ]}
      - [ ... ]  # list of events
    Returns a list of dict events (shallow-copied).
    """
    if isinstance(model_json, dict):
        if "all_events" in model_json and isinstance(model_json["all_events"], list):
            return [e.copy() for e in model_json["all_events"] if isinstance(e, dict)]
        if "events" in model_json and isinstance(model_json["events"], list):
            return [e.copy() for e in model_json["events"] if isinstance(e, dict)]
    if isinstance(model_json, list):
        return [e.copy() for e in model_json if isinstance(e, dict)]
    return []

def infer_instant_seconds(ev: dict) -> float | None:
    """
    Try to find a single representative timestamp for the event.
    Checks common fields the model may output.
    """
    keys = ["time_happened", "timestamp", "time", "ts", "event_time", "happened_at", "when"]
    for k in keys:
        if k in ev:
            sec = coerce_seconds(ev[k], default=None)
            if sec is not None:
                return sec

    # Search description for "at 00:12" or "at 10.5s"
    desc = str(ev.get("description", "")).lower()
    m = re.search(r"\bat\s+(\d{1,2}:\d{2}(?:\.\d{1,3})?)\b", desc)
    if m:
        sec = to_seconds_from_text(m.group(1), default=None)
        if sec is not None:
            return sec
    m2 = re.search(r"\bat\s+([0-9]+(?:\.[0-9]+)?)\s*s\b", desc)
    if m2:
        return float(m2.group(1))

    return None

def coerce_start_end_seconds(ev: dict) -> tuple[float, float]:
    """
    Ensure we have numeric start/end seconds for the event.
    Priority:
      1) If both provided (any type), coerce to seconds.
      2) Else if a single instant is present, build a window around it.
      3) Else default [0, DEFAULT_EVENT_LEN].
    """
    s_raw = ev.get("start_time", ev.get("startTime"))
    e_raw = ev.get("end_time", ev.get("endTime"))

    s = coerce_seconds(s_raw, default=None)
    e = coerce_seconds(e_raw, default=None)

    if s is not None and e is not None:
        start = max(0.0, s)
        end = max(start, e)
        return start, end

    t = infer_instant_seconds(ev)
    if t is not None:
        start = max(0.0, t - PRE_EVENT_SECONDS)
        end = t + POST_EVENT_SECONDS
        if CLIP_DURATION_SECONDS is not None:
            end = min(end, CLIP_DURATION_SECONDS)
        return start, max(start, end)

    start = 0.0
    end = start + DEFAULT_EVENT_LEN
    return start, end

def normalize_event(ev: dict) -> dict:
    """
    Returns a normalized event dict with:
      - start_time, end_time (floats, seconds)
      - startTime, endTime (MM:SS strings)
    Keeps other fields untouched.
    """
    out = ev.copy()
    start, end = coerce_start_end_seconds(out)
    out["start_time"] = float(start)
    out["end_time"] = float(end)
    out["startTime"] = to_mmss(start)
    out["endTime"] = to_mmss(end)
    return out

# ---------------- Build messages ----------------
def build_messages(video_path: str, system_prompt: str | None, user_prompt: str):
    """
    Construct a single-turn chat input for the VLM pipeline using frontend prompts.
    - If system_prompt is provided, include it first as text.
    - The user prompt is prefixed with "<video>\n" to indicate video input.
    """
    content = []
    if system_prompt:
        content.append({"type": "text", "text": system_prompt})
    # Ensure <video> tag precedes the user prompt
    content.append({"type": "text", "text": f"<video>\n{user_prompt}"})
    content.append({"type": "video", "video": video_path})
    return [{"role": "user", "content": content}]

# ---------------- Main flow ----------------
def main():
    args = parse_args()

    if not os.path.exists(args.video):
        print(f"Error: Video file not found at: {args.video}")
        sys.exit(1)

    print("Starting evaluation...")
    print(f"Loading base model from: {MODEL_DIR}")
    print(f"Loading adapter from: {ADAPTER_DIR}")

    # Processor (saved with final checkpoint)
    processor = AutoProcessor.from_pretrained(ADAPTER_DIR, trust_remote_code=True)
    if getattr(processor, "tokenizer", None) and processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # Base model (4-bit)
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    config = AutoConfig.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_DIR,
        config=config,
        quantization_config=quant_cfg,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Merge LoRA adapters
    print("Merging LoRA adapters into base model...")
    model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    print("Model loading and merging complete.")

    # Pipeline (IMPORTANT: pass processor= for VLMs)
    pipe = pipeline(
        "image-text-to-text",
        model=model,
        processor=processor,
        device_map="auto",
        trust_remote_code=True,
    )
    print("--- Pipeline created successfully. ---")

    # Messages with frontend-provided prompts
    messages = build_messages(args.video, args.system_prompt, args.user_prompt)
    print(f"Processing video: {args.video}")

    # Inference (deterministic to reduce rambling)
    output = pipe(
        messages,
        max_new_tokens=640,
        do_sample=False,
        temperature=0.0,
        repetition_penalty=1.0,
        stop_strings=["</answer>", "```"],  # harmless if unsupported
    )

    print("\n--- SUCCESS! INFERENCE COMPLETED. ---\n")
    print("Model's Raw Output:")

    # Extract assistant text
    try:
        assistant_reply = output[0]["generated_text"][-1]["content"]
        print(assistant_reply)
    except Exception as e:
        print(f"\nUnexpected pipeline output structure: {e}")
        print(output)
        sys.exit(1)

    # Extract JSON: prefer <answer> ... </answer>, else fenced, else balanced
    candidate = extract_answer_json(assistant_reply) or extract_fenced_json(assistant_reply) or extract_balanced_json(assistant_reply)

    # Try parse; if fail, auto-repair common issues and retry
    model_json = try_parse_json(candidate)
    if model_json is None:
        repaired = fix_common_json_issues(candidate)
        model_json = try_parse_json(repaired)

    if model_json is None:
        print("\nWarning: Could not parse JSON from model output even after repair.")
        sys.exit(1)

    # Normalize to list of events
    events = ensure_events_list(model_json)

    # Timing fixes per event
    fixed = [normalize_event(e) for e in events]

    # Emit both views:
    # (A) Pretty JSON with mm:ss strings
    norm_mmss = {"events": [{**{k: v for k, v in e.items() if k not in ("start_time", "end_time")}, "startTime": e["startTime"], "endTime": e["endTime"]} for e in fixed]}
    print("\n--- Normalized JSON (mm:ss) ---")
    print(json.dumps(norm_mmss, indent=2, ensure_ascii=False))

    # (B) Programmatic JSON with numeric seconds
    norm_seconds = {"events": fixed}
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(norm_seconds, f, ensure_ascii=False, indent=2)
        print(f"\nSaved JSON with seconds to: {args.json_out}")


if __name__ == "__main__":
    main()