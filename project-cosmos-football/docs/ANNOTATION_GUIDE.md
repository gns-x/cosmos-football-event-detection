
# 🎯 Manual Ground Truth Annotation Guide

## 📋 Annotation Format

Each video should have a corresponding JSON file with the following structure:

```json
[
  {
    "description": "Detailed description of the football action",
    "start_time": "0:1:32",
    "end_time": "0:1:38", 
    "event": "Goal"
  }
]
```

## 🏷️ Event Classes

1. **Goal** - Ball crosses goal line
2. **Penalty Shot** - Player takes penalty kick
3. **Red Card** - Referee shows red card
4. **Yellow Card** - Referee shows yellow card
5. **Corner Kick** - Corner kick being taken
6. **Free Kick** - Free kick being taken
7. **Throw In** - Player takes throw-in
8. **Offside** - Offside decision

## 📝 Description Guidelines

- **Be Specific**: Include player numbers, team colors, locations
- **Include Context**: What led to the event, what happened after
- **Use Football Terminology**: Proper football terms and phrases
- **Be Descriptive**: Paint a clear picture of the action

## ⏰ Time Format

- Use format: `MM:SS` or `H:MM:SS`
- Start time: When the action begins
- End time: When the action completes

## 🎯 Example Annotations

### Goal Example:
```json
{
  "description": "Player #10 (Messi) from PSG, in the blue jersey, curls a free-kick past the wall into the top left corner.",
  "start_time": "0:1:32",
  "end_time": "0:1:38",
  "event": "Goal"
}
```

### Yellow Card Example:
```json
{
  "description": "Player #7 (Ronaldo) from Al-Nassr, in the yellow jersey, is shown a yellow card for a late tackle on the defender.",
  "start_time": "0:2:45",
  "end_time": "0:2:51",
  "event": "Yellow Card"
}
```

## 🔍 Quality Checklist

- [ ] Description is detailed and specific
- [ ] Time stamps are accurate
- [ ] Event class is correct
- [ ] JSON format is valid
- [ ] All actions in video are captured
- [ ] No duplicate or missing annotations
