# Agentic Video Editing Pipeline - Quick Start

## What Is This?

An AI-powered video editing system that lets you edit multiple video clips using natural language commands like:
- "Remove from 1:23 to 2:45"
- "Cut where I say um too much"
- "Move clip 3 to the beginning"
- "Add MrBeast style captions"

## Installation

1. **Prerequisites**:
   - Python 3.8+
   - FFmpeg installed
   - Gemini API key (get from https://ai.google.dev/)

2. **Install dependencies** (if not already done):
   ```bash
   pip install google-genai transformers torch librosa pysubs2
   ```

3. **Set API key**:
   ```bash
   export GEMINI_API_KEY="your-api-key-here"
   ```

## Quick Test

```bash
# Verify everything is set up
python3 test_imports.py

# Should see:
# 🎉 All tests passed! The agentic pipeline is ready to use.
```

## Usage

### Basic Command

```bash
python3 editor_agent.py "video1.mov" "video2.mov" "video3.mov" --agentic
```

### What Happens

1. **Initial Processing** (automatic):
   - Stitches all clips together
   - Removes long pauses (1s+)
   - Transcribes the video
   - Runs AI quality check
   - Shows you a summary

2. **Interactive Editing**:
   - Type natural language commands
   - System applies edits
   - Re-transcribes after each change
   - Type 'done' when finished

3. **Final Export**:
   - Saves `output/final_video.mp4`
   - Preserves all intermediate edits
   - Saves session state for recovery

### Example Session

```
$ python3 editor_agent.py "assests/Ricky 1.mov" "assests/Ricky 2.mov" "assests/Ricky 3.mov" --agentic

=== INITIAL PROCESSING ===
1. Stitching 3 clips together...
2. Removing pauses (1.0s+ threshold)...
3. Transcribing video...
4. Running AI quality check...
✅ Initial processing complete!

Summary:
- Combined duration: 45.2s
- Clips: 3 clips
- Transcript: 127 words

=== AGENTIC EDITING MODE ===

>> Remove from 0:05 to 0:10
✂️  Cutting from 0:05 to 0:10...
📝 Re-transcribing...
✅ Cut applied successfully!

>> Move clip 2 to the beginning
📦 Moving clip 1 to position 0...
🔄 Re-stitching video...
✅ Clips reordered successfully!

>> Add captions in MrBeast style
🎨 Adding mrbeast captions...
✅ Captions added!

>> done
💾 Exporting final video...
✅ Final video saved: output/final_video.mp4
```

## Available Commands

### Time-Based Cuts
```
Remove from 1:23 to 2:45
Cut the first 10 seconds
Delete from 0:30 to 1:00
```

### Semantic Cuts
```
Remove where I say um
Cut the repetitions
Delete the mistakes
```

### Clip Reordering
```
Move clip 3 to the front
Put last clip first
```

### Quality Checks
```
Run quality check
Find and remove repetitions
```

### Captions
```
Add MrBeast style captions
Add captions in TikTok style
Add captions
```

### Utility
```
undo          # Revert last edit
help          # Show all commands
done          # Finish and export
```

## Options

```bash
python3 editor_agent.py video1.mov video2.mov --agentic [OPTIONS]

--output DIR              # Output directory (default: output)
--pause-threshold SEC     # Pause detection threshold (default: 1.0)
--skip-quality-check      # Skip initial AI quality check
```

## Output Files

After running, you'll find:

```
output/
├── working_video.mp4           # Initial processed video
├── edit_0.mp4                  # After first edit
├── edit_1.mp4                  # After second edit
├── edit_2.mp4                  # After third edit
├── final_video.mp4             # Final exported video
├── session_state.json          # Session state (for recovery)
└── *.json                      # Transcript files
```

## Troubleshooting

### "GEMINI_API_KEY not found"
```bash
export GEMINI_API_KEY="your-key-here"
```

### "FFmpeg not found"
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

### Slow transcription
The system uses GPU acceleration if available (CUDA/MPS). Check that you have the right torch version installed for your GPU.

### "No module named 'google.generativeai'"
```bash
pip install google-genai
```

## Tips

1. **Be specific**: "Remove from 1:23 to 2:45" is better than "cut the middle"
2. **Review before confirming**: Semantic cuts show you what they found before applying
3. **Use undo freely**: All edits are reversible (except captions)
4. **Add captions last**: They're hard to remove once burned in
5. **Check the transcript**: Quality of semantic cuts depends on transcription accuracy

## Documentation

- **User Guide**: `AGENTIC_MODE.md` - Detailed user documentation
- **Developer Guide**: `DEVELOPER_GUIDE.md` - For extending the system
- **Implementation**: `IMPLEMENTATION_SUMMARY.md` - Technical architecture

## Example Videos

Test with the included Ricky videos:
```bash
python3 editor_agent.py \
    "assests/Ricky 1.mov" \
    "assests/Ricky 2.mov" \
    "assests/Ricky 3.mov" \
    --agentic
```

## What Makes This "Agentic"?

1. **Natural Language Interface**: No need to learn commands - just describe what you want
2. **AI-Powered Understanding**: Gemini interprets your intent and executes the right actions
3. **Semantic Search**: Find content by meaning, not just timestamps
4. **Autonomous Quality Checks**: AI automatically finds and fixes issues
5. **Interactive Loop**: Iteratively refine until you're satisfied

## Performance

- **Initial processing**: 2-5 minutes for 3 short clips
- **Time-based cuts**: ~30 seconds
- **Semantic cuts**: 1-2 minutes (includes AI search + re-transcription)
- **Clip reordering**: 1-2 minutes (includes re-stitching + re-transcription)
- **Quality check**: 2-5 minutes
- **Captions**: 1-3 minutes

GPU acceleration significantly speeds up transcription.

## Limitations

- Can't preview edits before applying (coming soon)
- Re-transcription takes time after each edit
- Captions can't be easily removed once added
- Clip boundaries are approximate after edits

## Support

For issues or questions:
1. Check `AGENTIC_MODE.md` for detailed documentation
2. Review `DEVELOPER_GUIDE.md` for technical details
3. Run `python3 test_imports.py` to verify setup

## License

Same as the main project.

---

**Ready to start?**

```bash
export GEMINI_API_KEY="your-key"
python3 editor_agent.py video1.mov video2.mov video3.mov --agentic
```

Happy editing! 🎬
