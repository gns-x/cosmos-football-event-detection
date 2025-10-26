# 🎯 **Manual Ground Truth Annotation System**

## **📋 Overview**

This system creates detailed JSON ground truth annotations for each processed video, following the exact format specified for the Cosmos football video analysis project.

## **📁 File Structure**

```
03_annotation/
├── ground_truth_json/
│   ├── goal_01.json                    # Example: Detailed goal annotation
│   ├── penalty_shot_01.json           # Example: Penalty shot annotation
│   ├── red_card_01.json               # Example: Red card annotation
│   ├── yellow_card_01.json            # Example: Yellow card annotation
│   ├── corner_kick_01.json            # Example: Corner kick annotation
│   ├── free_kick_01.json              # Example: Free kick annotation
│   ├── throw_in_01.json               # Example: Throw-in annotation
│   ├── offside_01.json                # Example: Offside annotation
│   └── ANNOTATION_GUIDE.md            # Detailed annotation guide
├── create_ground_truth.py             # Automated annotation creator
├── manual_annotation.py               # Interactive manual annotation
└── setup_annotation_tool.py           # Web-based annotation tool
```

## **📝 JSON Format Specification**

### **Exact Format (as specified):**
```json
[
  {
    "description": "Player #10 (Messi) from PSG, in the blue jersey, curls a free-kick past the wall into the top left corner.",
    "start_time": "0:1:32",
    "end_time": "0:1:38",
    "event": "Goal"
  },
  {
    "description": "Player #7 (Ronaldo) from Al-Nassr, in the yellow jersey, is shown a yellow card for a late tackle on the defender.",
    "start_time": "0:2:45",
    "end_time": "0:2:51",
    "event": "Yellow Card"
  }
]
```

### **Extended Format (with additional metadata):**
```json
{
  "video_file": "goal_sample_1_processed.mp4",
  "class": "goal",
  "video_info": {
    "duration": 10.008005,
    "fps": "4/1",
    "duration_formatted": "0:10"
  },
  "annotations": [
    {
      "description": "Player scores a goal in the goal video. The ball crosses the goal line and the referee signals a goal.",
      "start_time": "0:00:05",
      "end_time": "0:00:15",
      "event": "Goal",
      "confidence": 1.0,
      "details": {
        "action": "goal_scored",
        "location": "goal_area",
        "outcome": "successful_goal"
      }
    }
  ],
  "created_date": "2025-10-24T22:18:02.872813",
  "annotator": "manual",
  "status": "completed"
}
```

## **🎯 Event Classes**

| Class | Description | Example Event |
|-------|-------------|---------------|
| **Goal** | Ball crosses goal line | "Player scores a goal" |
| **Penalty Shot** | Player takes penalty kick | "Player takes penalty shot" |
| **Red Card** | Referee shows red card | "Player shown red card" |
| **Yellow Card** | Referee shows yellow card | "Player cautioned with yellow card" |
| **Corner Kick** | Corner kick being taken | "Player takes corner kick" |
| **Free Kick** | Free kick being taken | "Player takes free kick" |
| **Throw In** | Player takes throw-in | "Player takes throw-in" |
| **Offside** | Offside decision | "Offside decision made" |

## **📝 Description Guidelines**

### **Be Specific and Detailed:**
- **Player Information**: Include player numbers, names, team colors
- **Location**: Specify field position, goal area, penalty spot, etc.
- **Action Details**: Describe the specific football action
- **Context**: What led to the event, what happened after
- **Outcome**: Result of the action

### **Example Descriptions:**

#### **Goal:**
```
"Player #10 (Messi) from PSG, in the blue jersey, curls a free-kick past the wall into the top left corner."
```

#### **Yellow Card:**
```
"Player #7 (Ronaldo) from Al-Nassr, in the yellow jersey, is shown a yellow card for a late tackle on the defender."
```

#### **Penalty Shot:**
```
"Player #9 steps up to take the penalty, places the ball on the spot, and strikes it into the bottom right corner past the diving goalkeeper."
```

## **⏰ Time Format**

- **Format**: `MM:SS` or `H:MM:SS`
- **Start Time**: When the action begins
- **End Time**: When the action completes
- **Examples**:
  - `"0:1:32"` (1 minute 32 seconds)
  - `"0:2:45"` (2 minutes 45 seconds)
  - `"0:00:05"` (5 seconds)

## **🛠️ Annotation Tools**

### **1. Automated Annotation Creator**
```bash
# Create basic annotations for all videos
python3 03_annotation/create_ground_truth.py
```

### **2. Interactive Manual Annotation**
```bash
# Annotate all videos interactively
python3 03_annotation/manual_annotation.py

# Annotate specific video
python3 03_annotation/manual_annotation.py --video "path/to/video.mp4" --class "goal"
```

### **3. Web-based Annotation Tool**
```bash
# Start web annotation interface
cd 03_annotation/annotation_tool
./launch.sh
# Open browser to: http://localhost:5000
```

## **📊 Current Status**

### **✅ Completed:**
- **24 Videos**: All sample videos processed
- **8 Classes**: All football action classes covered
- **JSON Format**: Exact format specification implemented
- **Annotation Tools**: Multiple annotation methods available

### **📁 Generated Files:**
- **24 JSON Files**: One per processed video
- **Annotation Guide**: Comprehensive documentation
- **Example Files**: goal_01.json with exact format
- **Interactive Tools**: Manual and automated annotation

## **🎯 Quality Checklist**

### **For Each Annotation:**
- [ ] **Description is detailed and specific**
- [ ] **Time stamps are accurate**
- [ ] **Event class is correct**
- [ ] **JSON format is valid**
- [ ] **All actions in video are captured**
- [ ] **No duplicate or missing annotations**

### **For Each Video:**
- [ ] **Video file exists and is playable**
- [ ] **JSON file created with correct name**
- [ ] **All events in video are annotated**
- [ ] **Time stamps match video content**
- [ ] **Descriptions are football-specific**

## **🚀 Usage Examples**

### **Create Basic Annotations:**
```bash
cd /Users/Genesis/Desktop/upwork/Nvidia-AI/project-cosmos-football
python3 03_annotation/create_ground_truth.py
```

### **Manual Annotation:**
```bash
# Interactive annotation for all videos
python3 03_annotation/manual_annotation.py

# Annotate specific video
python3 03_annotation/manual_annotation.py --video "02_preprocessing/processed_videos/goal/goal_sample_1_processed.mp4" --class "goal"
```

### **Web-based Annotation:**
```bash
cd 03_annotation/annotation_tool
./launch.sh
# Open http://localhost:5000 in browser
```

## **📋 Example Output Files**

### **goal_01.json:**
```json
[
  {
    "description": "Player #10 (Messi) from PSG, in the blue jersey, curls a free-kick past the wall into the top left corner.",
    "start_time": "0:1:32",
    "end_time": "0:1:38",
    "event": "Goal"
  },
  {
    "description": "Player #7 (Ronaldo) from Al-Nassr, in the yellow jersey, is shown a yellow card for a late tackle on the defender.",
    "start_time": "0:2:45",
    "end_time": "0:2:51",
    "event": "Yellow Card"
  }
]
```

## **🎯 Next Steps**

1. **Review Annotations**: Check quality of generated annotations
2. **Manual Refinement**: Use interactive tools for detailed annotation
3. **Quality Control**: Verify all videos are properly annotated
4. **Dataset Integration**: Use annotations for training dataset creation

---

**Status: ✅ Manual Annotation System Complete**
**Ready for: 🎯 Phase 3 - Training & Fine-tuning**
