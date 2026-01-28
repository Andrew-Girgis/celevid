# Developer Guide: Agentic Video Editing Pipeline

## Quick Start for Developers

### Running the System

```bash
# Activate virtual environment (if using one)
source .venv/bin/activate

# Set API key
export GEMINI_API_KEY="your-key-here"

# Run agentic mode
python3 editor_agent.py "assests/Ricky 1.mov" "assests/Ricky 2.mov" "assests/Ricky 3.mov" --agentic
```

### Adding New Commands

To add a new command type:

1. **Update CommandInterpreter** (`command_interpreter.py`):
   ```python
   # In _build_parsing_prompt(), add new action:
   5. "my_action" - Description
      Parameters: {"param1": type, "param2": type}
      Examples: "Do something"
   ```

2. **Add Handler** in `agentic_pipeline.py`:
   ```python
   def _handle_my_action(self, params: dict) -> bool:
       """Handle my new action."""
       try:
           # Your implementation here
           self.session.add_edit(
               "my_action",
               params,
               output_video_path
           )
           return True
       except Exception as e:
           print(f"Error: {e}")
           return False
   ```

3. **Route in _execute_command**:
   ```python
   elif action == 'my_action':
       return self._handle_my_action(command['parameters'])
   ```

### Understanding the Code Flow

#### 1. Command Parsing Flow

```
User: "Remove from 1:23 to 2:45"
    ↓
CommandInterpreter.parse_command()
    ↓
_build_parsing_prompt() - Creates Gemini prompt with context
    ↓
Gemini 2.0 Flash - Parses to JSON
    ↓
_validate_result() - Validates parameters
    ↓
Returns: {
    "action": "cut",
    "parameters": {"start_time": 83.0, "end_time": 165.0},
    "confidence": 0.95,
    "clarification_needed": false
}
```

#### 2. Edit Execution Flow

```
Parsed Command
    ↓
AgenticVideoEditor._execute_command()
    ↓
Route to specific handler (e.g., _handle_cut)
    ↓
Execute operation (VideoEditor, ContentEditor, etc.)
    ↓
Update video file (edit_N.mp4)
    ↓
Re-transcribe new video
    ↓
session.add_edit() - Record in history
    ↓
session.save_state() - Persist to JSON
    ↓
Return success to user
```

#### 3. State Management Flow

```
EditingSession initialized
    ↓
edit_history: []
clip_order: [0, 1, 2]
current_video: None
    ↓
After initial processing:
edit_history: [
    {type: "initial", video: "output/working_video.mp4", ...}
]
current_video: "output/working_video.mp4"
    ↓
After first cut:
edit_history: [
    {type: "initial", ...},
    {type: "cut", video: "output/edit_0.mp4", params: {...}}
]
current_video: "output/edit_0.mp4"
    ↓
On undo:
edit_history: [{type: "initial", ...}]  # Last entry removed
current_video: "output/working_video.mp4"  # Restored from previous
```

## Key Classes and Methods

### CommandInterpreter

**Purpose**: Convert natural language to structured actions

**Key Methods**:
```python
parse_command(user_input: str, context: dict) -> dict
    # Main entry point for command parsing
    # Returns action dict with confidence score

search_transcript_semantically(description: str, transcript: dict) -> List[Tuple]
    # AI-powered transcript search
    # Returns matching segments with timestamps

parse_timestamp(timestamp_str: str) -> Optional[float]
    # Parse various timestamp formats to seconds
    # Supports MM:SS, "Xs", "X minutes Y seconds"
```

**Usage Example**:
```python
interpreter = CommandInterpreter(api_key)
context = {
    "duration": 120.5,
    "num_clips": 3,
    "transcript_snippet": "I built this tool to help...",
    "clip_boundaries": [(0, 40), (40, 80), (80, 120)]
}
command = interpreter.parse_command("Remove from 1:23 to 2:45", context)
# Returns: {"action": "cut", "parameters": {...}, ...}
```

### EditingSession

**Purpose**: Manage state and enable undo

**Key Methods**:
```python
add_edit(edit_type: str, parameters: dict, video_path: str)
    # Record new edit in history
    # Updates current_video
    # Saves state to JSON

undo_last_edit() -> Optional[dict]
    # Remove last edit from history
    # Restore previous video/state
    # Returns previous edit record

get_context_dict() -> dict
    # Build context for command parsing
    # Returns duration, clips, transcript snippet, boundaries

save_state() / load_state()
    # Persist/restore session from JSON
```

**Usage Example**:
```python
session = EditingSession(video_paths, output_dir)
session.add_edit("cut", {"cuts": [(10, 20)]}, "output/edit_0.mp4")
session.undo_last_edit()  # Reverts to previous state
context = session.get_context_dict()  # For command parsing
```

### AgenticVideoEditor

**Purpose**: Main orchestrator for interactive editing

**Key Methods**:
```python
start_session(video_paths, output_dir, ...)
    # Initialize session
    # Run initial processing
    # Enter interactive loop

_interactive_loop()
    # Main command loop
    # Parse → Execute → Update → Repeat

_execute_command(command: dict) -> bool
    # Route to specific handler
    # Returns success/failure

_handle_cut(params) / _handle_move_clip(params) / etc.
    # Specific command handlers
    # Each updates session state
```

**Usage Example**:
```python
editor = AgenticVideoEditor()
editor.start_session(
    ["clip1.mov", "clip2.mov"],
    output_dir="output",
    pause_threshold=1.0
)
# Enters interactive loop automatically
```

## Extending the System

### Adding a New Caption Style

1. **Modify CaptionService** (`caption_service.py`):
   ```python
   def _get_style_config(self, style: str) -> dict:
       # Add new style
       elif style == "my_style":
           return {
               "fontsize": 48,
               "fontcolor": "#00FF00",
               "font": "Arial-Bold",
               # ... other properties
           }
   ```

2. **Update Documentation**:
   - Add to AGENTIC_MODE.md under "Add Captions"
   - Document in help text

### Adding a New Quality Check

1. **Modify ContentEditor** (`content_editor.py`):
   ```python
   # In _identify_cuts_with_gemini() prompt:
   """
   Look for:
   ...
   6. MY_NEW_CHECK: Description of what to look for
   """
   ```

2. **The ContentEditor will automatically use it**
   - No changes to agentic_pipeline.py needed
   - Returns same format: (start, end, reason) tuples

### Adding Session Recovery

Currently state is saved but not loaded on startup. To enable:

```python
# In AgenticVideoEditor.start_session():
def start_session(self, video_paths, output_dir, resume=False):
    if resume:
        # Try loading existing session
        self.session = EditingSession.load_state(output_dir)
        if self.session:
            print("Resumed previous session")
            self._interactive_loop()
            return

    # Otherwise create new session
    self.session = EditingSession(video_paths, output_dir)
    # ... rest of initialization
```

## Debugging Tips

### Enable Verbose Logging

Add to top of files:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Inspect Session State

```python
# During interactive loop:
>> help  # Pause and inspect
# In another terminal:
cat output/session_state.json | python3 -m json.tool
```

### Test Command Parsing Independently

```python
from command_interpreter import CommandInterpreter
interpreter = CommandInterpreter(api_key)

context = {"duration": 120, "num_clips": 3, "transcript_snippet": "test"}
result = interpreter.parse_command("Remove from 1:23 to 2:45", context)
print(json.dumps(result, indent=2))
```

### Verify Video Files

```bash
# Check edit history
ls -lh output/edit_*.mp4

# Check video duration
ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 output/edit_0.mp4
```

## Common Issues & Solutions

### Issue: "GEMINI_API_KEY not found"
**Solution**: Export the environment variable:
```bash
export GEMINI_API_KEY="your-key-here"
```

### Issue: Transcription is slow
**Solution**: Check if GPU is being used:
```python
# In transcribe_video(), should print:
🚀 Using NVIDIA GPU (CUDA)  # or MPS for Apple
```
If seeing CPU warning, install torch with GPU support.

### Issue: FFmpeg not found
**Solution**: Install FFmpeg:
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

### Issue: Semantic search returns nothing
**Solution**: Check transcript format:
```python
# Transcript must have 'chunks' with timestamps
{
    "text": "...",
    "chunks": [
        {"text": "word", "timestamp": [0.0, 0.5]},
        ...
    ]
}
```

### Issue: Undo doesn't work
**Solution**: Can't undo initial processing (by design). Need at least one edit first.

## Performance Optimization

### Reduce Re-transcription Time

Option 1: Cache for simple operations
```python
def _handle_add_captions(self, params):
    # Don't re-transcribe - captions don't change audio
    # Just update video path
    pass
```

Option 2: Use smaller Whisper model
```python
# In transcribe_video(), change model:
model="distil-whisper/distil-small.en"  # Faster, less accurate
```

### Batch Processing

Currently processes videos sequentially. To parallelize:
```python
from concurrent.futures import ProcessPoolExecutor

with ProcessPoolExecutor() as executor:
    futures = [executor.submit(transcribe_video, path) for path in paths]
    results = [f.result() for f in futures]
```

## Testing

### Unit Tests (Example)

```python
# test_command_interpreter.py
import unittest
from command_interpreter import CommandInterpreter

class TestCommandInterpreter(unittest.TestCase):
    def setUp(self):
        self.interpreter = CommandInterpreter(api_key)

    def test_parse_timestamp(self):
        self.assertEqual(self.interpreter.parse_timestamp("1:23"), 83.0)
        self.assertEqual(self.interpreter.parse_timestamp("83s"), 83.0)

    def test_parse_cut_command(self):
        context = {"duration": 120, "num_clips": 2, ...}
        result = self.interpreter.parse_command("Remove from 1:00 to 2:00", context)
        self.assertEqual(result['action'], 'cut')
        self.assertEqual(result['parameters']['start_time'], 60.0)
```

### Integration Tests

```bash
# test_agentic_pipeline.sh
#!/bin/bash
set -e

# Run with test videos
python3 editor_agent.py test1.mov test2.mov --agentic <<EOF
Remove from 0:00 to 0:05
undo
done
EOF

# Verify output exists
test -f output/final_video.mp4
echo "Test passed!"
```

## Code Style

Follow existing patterns:
- Docstrings for all public methods
- Type hints for parameters and returns
- Error handling with try/except
- Print user-facing messages with emoji icons
- Use Path for file operations
- Validate inputs early

## Contributing

When adding features:
1. Update relevant .md documentation
2. Add docstrings to new methods
3. Test manually with real videos
4. Check syntax with `python3 -m py_compile`
5. Update IMPLEMENTATION_SUMMARY.md

## Resources

- FFmpeg documentation: https://ffmpeg.org/ffmpeg.html
- Gemini API: https://ai.google.dev/docs
- Whisper models: https://huggingface.co/distil-whisper
- pysubs2: https://github.com/tkarabela/pysubs2
