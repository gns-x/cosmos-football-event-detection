# football_analysis.py

from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

# Path to the downloaded model (or Hugging Face repo ID to auto-download)
MODEL_PATH = "./cosmos-reason1-7b"  # Or "nvidia/Cosmos-Reason1-7B"

# Load the model with BF16 for efficiency (adjust dtype if VRAM is limited)
llm = LLM(
    model=MODEL_PATH,
    dtype="bfloat16",  # Use "float16" if BF16 unsupported
    limit_mm_per_prompt={"image": 10, "video": 10},
)

# Sampling parameters for generation
sampling_params = SamplingParams(
    temperature=0.2,           # Lower temp for more deterministic JSON
    top_p=0.9,
    repetition_penalty=1.05,
    max_tokens=2048,           # Usually sufficient for structured outputs
)

# Football analysis system prompt: JSON-only, clear schema, no chain-of-thought
SYSTEM_PROMPT = """
You are an expert football (soccer) video analysis assistant.

Task:
- Detect football events from the provided video frames.
- Return ONLY valid JSON. Do not include any extra text, prose, or explanations.
- If uncertain, include "confidence": a float between 0 and 1.

Event taxonomy (type):
- "kickoff"
- "goal"
- "shot_on_target"
- "shot_off_target"
- "save"
- "foul"
- "offside"
- "card_yellow"
- "card_red"
- "corner"
- "free_kick"
- "penalty"
- "throw_in"
- "substitution"

Required JSON schema:
{
  "match_context": {
    "home_team": "string | null",
    "away_team": "string | null",
    "half": "1 | 2 | null",
    "stadium": "string | null"
  },
  "events": [
    {
      "type": "string (one of taxonomy above)",
      "timestamp": {
        "video_seconds": "number (float seconds from start)",
        "match_clock": "string | null (e.g., 12:34)"
      },
      "teams": {
        "attacking": "string | null",
        "defending": "string | null"
      },
      "players": [
        {
          "name": "string | null",
          "jersey_number": "number | null",
          "team": "string | null",
          "role": "string | null"  // e.g., "striker", "goalkeeper", "referee"
        }
      ],
      "ball": {
        "visible": "boolean",
        "position_px": { "x": "number | null", "y": "number | null" }  // frame pixel coords
      },
      "location": {
        "region": "string | null"  // e.g., "penalty_area_home", "center_circle", "left_flank_attack"
      },
      "outcome": "string | null",   // e.g., "goal_scored", "blocked", "saved", "missed", "foul_committed"
      "confidence": "number (0.0-1.0)"
    }
  ]
}

Guidelines:
- Use "null" when information cannot be reliably inferred.
- Use "video_seconds" based on frame index / fps if explicit clock is not visible.
- Do not invent player names or teams unless clearly visible (else set null).
- Prefer concise events over verbose notes.
- Return an empty "events" array if nothing is detected.
- Output MUST be a single JSON object conforming to the schema above.
"""

# Example messages: JSON-only output, video provided
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": [
        {"type": "text", "text": "Analyze this football clip. Detect events and output JSON strictly following the schema."},
        {"type": "video", "video": "./01_data_collection/raw_videos/goal", "fps": 4}  # keep fps=4
    ]}
]

# Process inputs
processor = AutoProcessor.from_pretrained(MODEL_PATH)
prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

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

# Generate output
outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
generated_text = outputs[0].outputs[0].text

print(generated_text)
