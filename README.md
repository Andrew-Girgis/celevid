# 🎬 Celevid - AI Video Editor

Automatically transcribe, remove pauses, and add captions to videos using AI.

## Features

- 🎤 **GPU-Accelerated Transcription** - Word-level timestamps using Distil-Whisper
- ✂️ **Smart Pause Removal** - Automatically detect and remove long pauses
- 📝 **Auto-Captions** - Multiple caption styles (TikTok, MrBeast, Professional)
- 🎥 **Quality Presets** - 4 encoding presets (fast, balanced, quality, maximum)
- 🎵 **Audio Extraction** - Extract and keep M4A audio for reuse/storage
- 📂 **Universal Format Support** - Accepts MOV, MP4, M4A, and more
- 🚀 **Fast Processing** - Uses Mac GPU (MPS) for transcription
- 📊 **Detailed Analytics** - File size reduction, time saved

## Installation

```bash
# Install dependencies
uv sync

# Make sure you have FFmpeg with libass
brew install ffmpeg-full
brew link ffmpeg-full --force
```

## Quick Start

### Basic Usage

```bash
# Process video with default settings (remove pauses + add TikTok captions)
uv run python process_video.py video.mp4
```

### Advanced Usage

```bash
# Custom caption style
uv run python process_video.py video.mp4 --style mrbeast

# Skip pause removal
uv run python process_video.py video.mp4 --no-pauses

# Skip captions
uv run python process_video.py video.mp4 --no-captions

# Adjust pause threshold (in seconds)
uv run python process_video.py video.mp4 --threshold 1.5

# More words per caption
uv run python process_video.py video.mp4 --words 5

# Custom output directory
uv run python process_video.py video.mp4 --output my_videos

# High quality encoding
uv run python process_video.py video.mp4 --quality quality

# Maximum quality (slower but best output)
uv run python process_video.py video.mp4 --quality maximum

# Skip audio extraction
uv run python process_video.py video.mp4 --no-audio
```

## Quality Presets

### Fast
- **Speed**: Fastest
- **Quality**: Good for drafts
- **Use case**: Quick previews, testing

### Balanced (Default)
- **Speed**: Fast
- **Quality**: Great for most uses
- **Use case**: Social media, general sharing

### Quality
- **Speed**: Medium
- **Quality**: High quality
- **Use case**: Professional content, uploads

### Maximum
- **Speed**: Slow
- **Quality**: Best possible
- **Use case**: Archival, final masters
```

## Caption Styles

### TikTok/Reels (Default)
- Bold white text
- Bottom center
- Modern, mobile-friendly

### MrBeast
- Large yellow text
- High impact
- Perfect for viral content

### Professional
- Clean, subtle
- Semi-transparent background
- Corporate/presentation style

### Default
- Simple white text
- Standard subtitles

## Output

The tool generates:
- `{name}_final.mp4` - Processed video with captions
- `{name}_audio.m4a` - Extracted audio (high quality, for reuse/storage)
- `{name}_transcript.json` - Full transcript with timestamps
- `{name}_edited.mp4` - Intermediate edit (if pauses removed)

### File Workflow
1. **Accepts any format**: MOV, MP4, M4A, AVI, etc.
2. **Extracts M4A**: High-quality audio stored for reuse
3. **Transcribes**: Uses M4A or original file
4. **Processes**: Removes pauses, adds captions with quality encoding
5. **Keeps M4A**: Perfect for re-processing or archival

## Project Structure

```
Celevid/
├── process_video.py      # Main consolidated pipeline
├── audio_processor.py    # Audio extraction & quality presets
├── caption_service.py    # Caption generation
├── video_editor.py       # Pause detection/removal
├── editor_agent.py       # Interactive editing agent
├── main.py              # Standalone transcription
├── output/              # Processed videos
│   ├── *_final.mp4      # Final captioned video
│   ├── *_audio.m4a      # Extracted audio
│   └── *_transcript.json # Transcripts
└── assets/              # Sample videos
```

## Examples

### Example 1: Social Media Ready
```bash
uv run python process_video.py my_video.mov --style tiktok --words 3
```
**Result**: Video with short, punchy captions optimized for TikTok/Reels

### Example 2: Long-form Content
```bash
uv run python process_video.py podcast.mp4 --threshold 2.0 --words 5 --style professional
```
**Result**: Removes 2s+ pauses, adds professional-style captions with longer phrases

### Example 3: Transcription Only
```bash
uv run python process_video.py interview.mov --no-pauses --no-captions
```
**Result**: Just generates transcript JSON

## Performance

- **Transcription**: ~1-2 minutes per minute of video (GPU-accelerated)
- **Pause Removal**: ~30 seconds per video
- **Caption Burning**: ~1 minute per video

**Example**: 15-second video → ~2 minutes total processing

## Requirements

- Python 3.12+
- FFmpeg with libass support
- Mac with Apple Silicon (for GPU acceleration)
- ~1.5GB disk space for AI model (one-time download)

## Troubleshooting

### FFmpeg subtitle error
```bash
# Install ffmpeg-full instead of regular ffmpeg
brew uninstall ffmpeg
brew install ffmpeg-full
brew link ffmpeg-full --force
```

### Slow transcription
- First run downloads 1.5GB model (one-time)
- Check GPU is being used: Look for "Device set to use mps"
- CPU fallback is 3-5x slower

### Memory issues
- Process shorter videos (<5 minutes)
- Close other GPU-intensive apps

## Advanced: Interactive Agent

For more control, use the interactive agent:

```bash
uv run python editor_agent.py video.mp4
```

This provides:
- Filler word detection
- Repetition analysis
- Interactive editing choices
- Preview before applying edits

## License

MIT

## Credits

Built with:
- [Distil-Whisper](https://github.com/huggingface/distil-whisper) for transcription
- [FFmpeg](https://ffmpeg.org/) for video processing
- [pysubs2](https://github.com/tkarabela/pysubs2) for subtitle generation
