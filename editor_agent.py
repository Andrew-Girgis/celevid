"""
Agentic Video Editor - Combines transcription with intelligent editing
"""
import sys
import json
from pathlib import Path
from transformers import pipeline
import torch
import librosa
from video_editor import VideoEditor


class TranscriptAnalyzer:
    """Analyzes transcript to detect editing opportunities."""

    def __init__(self, transcript_data: dict):
        self.words = transcript_data.get("chunks", [])
        self.text = transcript_data.get("text", "")

    def detect_filler_words(self) -> list:
        """Find filler words like um, uh, like, you know, etc."""
        fillers = {"um", "uh", "like", "you know", "so", "actually", "basically", "literally"}
        filler_segments = []

        for chunk in self.words:
            word = chunk["text"].strip().lower()
            if word in fillers:
                filler_segments.append({
                    "type": "filler",
                    "word": word,
                    "start": chunk["timestamp"][0],
                    "end": chunk["timestamp"][1],
                })

        return filler_segments

    def detect_repetitions(self) -> list:
        """Find repeated words/phrases."""
        repetitions = []
        prev_word = None

        for i, chunk in enumerate(self.words):
            word = chunk["text"].strip().lower()
            if word == prev_word and word not in {",", ".", "!", "?"}:
                repetitions.append({
                    "type": "repetition",
                    "word": word,
                    "start": self.words[i-1]["timestamp"][0],
                    "end": chunk["timestamp"][1],
                })
            prev_word = word

        return repetitions

    def find_long_sentences(self, threshold: float = 15.0) -> list:
        """Find sentences that run too long without pauses."""
        # This is a simple heuristic - could be improved
        long_segments = []
        sentence_start = None
        last_end = 0

        for chunk in self.words:
            word = chunk["text"].strip()
            if sentence_start is None:
                sentence_start = chunk["timestamp"][0]

            # End of sentence markers
            if word.endswith(('.', '!', '?')):
                duration = chunk["timestamp"][1] - sentence_start
                if duration > threshold:
                    long_segments.append({
                        "type": "long_sentence",
                        "start": sentence_start,
                        "end": chunk["timestamp"][1],
                        "duration": duration,
                    })
                sentence_start = None

        return long_segments


def transcribe_video(video_path: str) -> dict:
    """Transcribe video with word-level timestamps."""
    print(f"Transcribing {video_path}...")

    # Auto-detect best device: CUDA (NVIDIA GPU) > MPS (Apple Metal) > CPU
    if torch.cuda.is_available():
        device = "cuda"
        print("🚀 Using NVIDIA GPU (CUDA)")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("🚀 Using Apple Metal GPU (MPS)")
    else:
        device = "cpu"
        print("⚠️  Using CPU (slower)")

    pipe = pipeline(
        "automatic-speech-recognition",
        model="distil-whisper/distil-large-v3",
        torch_dtype=torch.float16,
        device=device,
    )

    # Load and trim audio
    audio_array, sampling_rate = librosa.load(video_path, sr=16000)
    audio_trimmed, trim_indices = librosa.effects.trim(audio_array, top_db=30)
    start_offset = trim_indices[0] / sampling_rate

    # Transcribe
    result = pipe(audio_trimmed, return_timestamps="word")

    # Adjust timestamps
    if result.get("chunks"):
        for chunk in result["chunks"]:
            if chunk["timestamp"][0] is not None:
                chunk["timestamp"] = (
                    chunk["timestamp"][0] + start_offset,
                    chunk["timestamp"][1] + start_offset if chunk["timestamp"][1] else None,
                )

    return result


def main():
    # Check for agentic mode
    if '--agentic' in sys.argv:
        from agentic_pipeline import AgenticVideoEditor

        # Extract video paths (non-flag arguments)
        video_paths = [arg for arg in sys.argv[1:] if not arg.startswith('--') and Path(arg).exists()]

        if not video_paths:
            print("Usage: python editor_agent.py video1.mp4 video2.mp4 [video3.mp4 ...] --agentic")
            print("\nOptions:")
            print("  --agentic             Enable agentic multi-clip editing mode")
            print("  --output DIR          Output directory (default: output)")
            print("  --pause-threshold SEC Pause detection threshold (default: 1.0)")
            print("  --skip-quality-check  Skip initial AI quality check")
            sys.exit(1)

        # Parse additional options
        output_dir = "output"
        pause_threshold = 1.0
        skip_quality_check = '--skip-quality-check' in sys.argv

        if '--output' in sys.argv:
            idx = sys.argv.index('--output')
            if idx + 1 < len(sys.argv):
                output_dir = sys.argv[idx + 1]

        if '--pause-threshold' in sys.argv:
            idx = sys.argv.index('--pause-threshold')
            if idx + 1 < len(sys.argv):
                pause_threshold = float(sys.argv[idx + 1])

        # Start agentic session
        try:
            editor = AgenticVideoEditor()
            editor.start_session(
                video_paths,
                output_dir=output_dir,
                pause_threshold=pause_threshold,
                skip_initial_quality_check=skip_quality_check
            )
        except KeyboardInterrupt:
            print("\n\n⚠️  Session interrupted")
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

        return

    # Original single-video mode
    if len(sys.argv) < 2:
        print("Usage: python editor_agent.py <video_file>")
        print("   or: python editor_agent.py video1.mp4 video2.mp4 [...] --agentic")
        sys.exit(1)

    video_path = sys.argv[1]
    video_name = Path(video_path).stem

    print("="*70)
    print("🎬 AGENTIC VIDEO EDITOR")
    print("="*70)

    # Step 1: Transcribe
    transcript = transcribe_video(video_path)

    # Save transcript
    transcript_path = f"assets/{video_name}_transcript.json"
    with open(transcript_path, 'w') as f:
        json.dump(transcript, f, indent=2)
    print(f"✓ Transcript saved: {transcript_path}")

    # Step 2: Analyze transcript
    print("\n" + "="*70)
    print("📊 ANALYZING TRANSCRIPT")
    print("="*70)

    analyzer = TranscriptAnalyzer(transcript)

    print(f"\n📝 Transcript:\n{transcript['text']}\n")

    # Detect issues
    fillers = analyzer.detect_filler_words()
    repetitions = analyzer.detect_repetitions()

    print(f"\n🔍 Found {len(fillers)} filler words")
    if fillers[:5]:  # Show first 5
        for filler in fillers[:5]:
            print(f"  - '{filler['word']}' at {filler['start']:.2f}s")

    print(f"\n🔄 Found {len(repetitions)} repetitions")
    if repetitions[:5]:
        for rep in repetitions[:5]:
            print(f"  - '{rep['word']}' repeated at {rep['start']:.2f}s")

    # Step 3: Detect pauses
    print("\n" + "="*70)
    print("⏸️  DETECTING LONG PAUSES")
    print("="*70)

    editor = VideoEditor(video_path, pause_threshold=1.0)
    pauses = editor.detect_silence()

    if pauses:
        print(f"\n⚠️  Found {len(pauses)} long pauses:")
        for i, (start, end, duration) in enumerate(pauses, 1):
            print(f"  {i}. {start:.2f}s - {end:.2f}s ({duration:.1f}s pause)")

    # Step 4: Create annotated preview
    print("\n" + "="*70)
    print("🎥 CREATING PREVIEW")
    print("="*70)

    if pauses:
        preview_path = f"assets/{video_name}_preview.mp4"
        editor.create_annotated_preview(preview_path)
        print(f"\n✓ Preview created with pause annotations")

    # Step 5: Show summary and options
    print("\n" + "="*70)
    print("📋 EDITING SUMMARY")
    print("="*70)

    total_issues = len(fillers) + len(repetitions) + len(pauses)
    print(f"\nTotal issues found: {total_issues}")
    print(f"  - {len(pauses)} long pauses")
    print(f"  - {len(fillers)} filler words")
    print(f"  - {len(repetitions)} repetitions")

    if pauses:
        total_pause_time = sum(d for _, _, d in pauses)
        print(f"\nPotential time savings: {total_pause_time:.1f}s")

        print("\n" + "="*70)
        print("⚡ QUICK ACTIONS")
        print("="*70)
        print("1. Remove all long pauses")
        print("2. Add captions to video")
        print("3. Add captions + remove pauses")
        print("4. Export transcript only")
        print("5. Exit")

        choice = input("\nSelect action [1-5]: ").strip()

        if choice == "1":
            final_path = f"assets/{video_name}_edited.mp4"
            editor.apply_edits(final_path)
        elif choice == "2":
            add_captions_to_video(video_path, transcript, video_name)
        elif choice == "3":
            # Edit video first, then add captions
            edited_path = f"assets/{video_name}_edited.mp4"
            editor.apply_edits(edited_path)
            add_captions_to_video(edited_path, transcript, f"{video_name}_edited")
        elif choice == "4":
            print(f"Transcript already saved to: {transcript_path}")
    else:
        print("\n✅ Video is already clean - no long pauses detected!")
        print(f"📄 Transcript saved to: {transcript_path}")

        # Still offer captions
        print("\n" + "="*70)
        print("⚡ ADDITIONAL OPTIONS")
        print("="*70)
        print("1. Add captions to video")
        print("2. Exit")

        choice = input("\nSelect action [1-2]: ").strip()

        if choice == "1":
            add_captions_to_video(video_path, transcript, video_name)


def add_captions_to_video(video_path: str, transcript: dict, video_name: str):
    """Add captions to video with style selection."""
    from caption_service import CaptionService

    print("\n" + "="*70)
    print("🎬 CAPTION STYLES")
    print("="*70)
    print("1. TikTok/Reels (Bold, modern)")
    print("2. MrBeast (Yellow, impact)")
    print("3. Professional (Clean, subtle)")
    print("4. Default (Simple)")

    style_choice = input("\nSelect style [1-4]: ").strip()
    style_map = {"1": "tiktok", "2": "mrbeast", "3": "professional", "4": "default"}
    style = style_map.get(style_choice, "tiktok")

    print(f"\nAdding {style} style captions...")

    caption_service = CaptionService()
    output_path = f"assets/{video_name}_captioned.mp4"

    try:
        caption_service.add_captions(
            video_path=video_path,
            transcript=transcript,
            output_path=output_path,
            style=style,
            words_per_caption=3
        )
        print(f"\n✅ Captions added successfully!")
        print(f"📹 Output: {output_path}")
    except Exception as e:
        print(f"\n❌ Error adding captions: {e}")


if __name__ == "__main__":
    main()
