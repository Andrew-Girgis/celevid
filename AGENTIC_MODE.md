# Agentic Video Editing Mode

## Overview

The agentic video editing mode allows you to stitch multiple video clips together and edit them using natural language commands. The system uses AI (Gemini) to understand your editing requests and apply them automatically.

## Quick Start

### Basic Usage

```bash
python editor_agent.py "assests/Ricky 1.mov" "assests/Ricky 2.mov" "assests/Ricky 3.mov" --agentic
```

### Initial Processing

When you start an agentic session, the system automatically:

1. **Stitches clips together** - Combines all input videos into one
2. **Removes pauses** - Detects and removes long silences (1.0s+ by default)
3. **Transcribes video** - Generates word-level transcript
4. **Runs AI quality check** - Identifies and removes repetitions, filler words, and mistakes

After processing, you'll see a summary:
```
Summary:
- Combined duration: 45.2s (was 67.8s, saved 22.6s)
- Clips: 3 clips
- Transcript: 127 words
```

## Natural Language Commands

### Time-Based Cuts

Remove specific time ranges from your video:

```
>> Remove from 1:23 to 2:45
>> Cut the first 10 seconds
>> Delete from 0:30 to 1:00
```

Supported time formats:
- `1:23` (MM:SS)
- `83s` (seconds)
- `1 minute 23 seconds` (natural language)

### Semantic Cuts

Remove content based on what was said:

```
>> Remove where I say um
>> Cut the repetitions
>> Delete the mistakes
>> Remove filler words
```

The system will:
1. Search the transcript for matching segments
2. Show you what it found
3. Ask for confirmation before cutting

### Clip Reordering

Change the order of your original clips:

```
>> Move clip 3 to the front
>> Put last clip first
>> Move clip 2 to position 0
```

Note: Clips are numbered starting from 1 in commands, but internally use 0-based indexing.

### Quality Operations

Run AI quality checks to find and fix issues:

```
>> Run quality check
>> Find and remove repetitions
```

The AI will analyze your transcript and identify:
- Repetitions (restarted sentences)
- Self-corrections ("wait", "let me start over")
- Excessive filler words
- False starts and mistakes

### Add Captions

Burn captions into your video with different styles:

```
>> Add MrBeast style captions
>> Add captions in TikTok style
>> Burn captions
```

Available styles:
- `mrbeast` - Yellow, impact font
- `tiktok` - Bold, modern style
- `default` - Simple, clean

### Other Commands

```
>> undo          # Revert to previous edit
>> help          # Show all available commands
>> done          # Finish and export final video
```

## Advanced Options

### Custom Output Directory

```bash
python editor_agent.py video1.mov video2.mov video3.mov --agentic --output my_edits
```

### Adjust Pause Threshold

Change how long a silence must be to be considered a pause (default: 1.0 seconds):

```bash
python editor_agent.py video1.mov video2.mov --agentic --pause-threshold 0.5
```

### Skip Initial Quality Check

Skip the AI quality analysis during initial processing:

```bash
python editor_agent.py video1.mov video2.mov --agentic --skip-quality-check
```

## Example Session

```bash
$ python editor_agent.py "assests/Ricky 1.mov" "assests/Ricky 2.mov" "assests/Ricky 3.mov" --agentic

=== INITIAL PROCESSING ===
1. Stitching 3 clips together...
2. Removing pauses (1.0s+ threshold)...
3. Transcribing video...
4. Running AI quality check...
   Found 2 repetitions, removing...
5. Processing complete!

Summary:
- Combined duration: 45.2s (was 67.8s, saved 22.6s)
- Clips: 3 clips
- Transcript: 127 words

=== AGENTIC EDITING MODE ===
Tell me what you'd like to edit. Examples:
  - "Remove from 1:23 to 2:45"
  - "Cut the part where I say um too much"
  - "Move clip 3 to the beginning"
  - "Add captions in MrBeast style"

Type 'help' for more commands, 'done' to finish.

>> Remove the part where I say um too much
🔍 Searching transcript for: um too much
Found 2 segments:
  1. 12.3s - 14.1s: "um um like you know"
  2. 28.7s - 29.9s: "um um um"
Remove all these segments? [y/n]: y
✂️  Applying cuts...
📝 Re-transcribing...
✅ Cut applied successfully!

>> Move clip 2 to the beginning
📦 Moving clip 1 to position 0...
🔄 Re-stitching video...
📝 Re-transcribing...
✅ Clips reordered successfully!

>> Add captions in MrBeast style
🎨 Adding mrbeast captions...
✅ Captions added!

>> done
💾 Exporting final video...
✅ Final video saved: output/final_video.mp4

Session: 20260127_223045
Clips: 3 clips
Duration: 38.5s (0:38)
Edits applied: 3
Current video: output/edit_2.mp4

Session state saved to: output/session_state.json
```

## Session State & Recovery

All edits are tracked in `output/session_state.json`. This allows you to:

- **Undo** any edit and revert to a previous state
- **Resume** a session if it crashes (future feature)
- **Track** the complete edit history

Each edit creates a new video file (`edit_0.mp4`, `edit_1.mp4`, etc.) so you never lose work.

## Architecture

### Components

1. **CommandInterpreter** (`command_interpreter.py`)
   - Parses natural language commands using Gemini
   - Converts commands to structured actions
   - Handles timestamp parsing and semantic search

2. **SessionState** (`session_state.py`)
   - Manages editing state across iterations
   - Tracks clip order, edit history, and current video
   - Enables undo functionality

3. **AgenticVideoEditor** (`agentic_pipeline.py`)
   - Main orchestrator for the interactive loop
   - Routes commands to appropriate handlers
   - Coordinates video processing and transcription

### Integration

The agentic mode leverages existing components:

- `process_multiple_videos()` - Initial stitching and processing
- `ContentEditor` - AI quality analysis with Gemini
- `VideoEditor.apply_content_cuts()` - Applying time-based cuts
- `CaptionService` - Burning captions with different styles
- `transcribe_video()` - Generating word-level transcripts

## Technical Details

### Gemini Integration

Commands are parsed using Gemini 2.0 Flash with structured JSON output:

```python
{
  "action": "cut",
  "parameters": {"start_time": 83.0, "end_time": 165.0},
  "confidence": 0.95,
  "clarification_needed": false
}
```

For semantic searches, Gemini analyzes the transcript and returns matching segments.

### Video Processing Pipeline

1. **FFmpeg concat demuxer** - Stitches clips efficiently
2. **Pause detection** - Uses librosa to find silences
3. **Transcription** - Distil-Whisper for fast, accurate transcripts
4. **Content analysis** - Gemini identifies quality issues
5. **Caption rendering** - FFmpeg burns captions with custom styles

## Tips

- Be specific in your commands for better results
- Review semantic cuts before confirming - the AI might find more than expected
- Use `undo` freely - all edits are reversible
- The system re-transcribes after each edit for accuracy
- Captions should be added last (they can't be undone easily)

## Limitations

- Semantic search depends on transcript quality
- Re-transcription takes time after each edit
- Clip boundaries are approximated after edits
- Caption addition is not easily reversible

## Future Improvements

- Session recovery from saved state
- Better clip boundary tracking
- Preview before applying edits
- Batch commands
- Custom caption styling
