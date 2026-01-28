"""
Celevid - AI Video Editor
Transcribe, edit pauses, and add captions to videos
"""
import sys
import json
import tempfile
from pathlib import Path
from transformers import pipeline
import torch
import librosa
from video_editor import VideoEditor
from caption_service import CaptionService
from audio_processor import AudioProcessor, VideoQualityPresets, process_video_with_quality, normalize_video_to_mp4
from transcript_cleaner import TranscriptCleaner
from content_editor_v2 import ContentEditorV2


def transcribe_video(video_path: str, output_dir: Path, audio_path: str = None) -> dict:
    """
    Transcribe video with word-level timestamps.

    Args:
        video_path: Original video file
        output_dir: Output directory
        audio_path: Pre-extracted audio file (M4A), if available

    Returns:
        Transcript dict with chunks
    """
    print(f"\n📝 Transcribing {Path(video_path).name}...")

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
        dtype=torch.float16,
        device=device,
    )

    # Use provided audio or extract from video
    source_audio = audio_path if audio_path else video_path

    # Load and trim audio (librosa handles format conversion internally)
    audio_array, sampling_rate = librosa.load(source_audio, sr=16000)
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

    # Save transcript
    transcript_path = output_dir / f"{Path(video_path).stem}_transcript.json"
    with open(transcript_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"✓ Transcript: {result['text']}")
    print(f"✓ Saved to: {transcript_path}")

    return result


def process_video(
    video_path: str,
    output_dir: str = "output",
    remove_pauses: bool = True,
    pause_threshold: float = 1.0,
    pause_buffer: float = 0.5,
    add_captions: bool = True,
    caption_style: str = "tiktok",
    caption_mode: str = "word_by_word",
    words_per_caption: int = 3,
    quality: str = "balanced",
    extract_audio: bool = True,
    normalize: bool = True,
    clean_transcript: bool = True,
    content_edit: bool = True
):
    """
    Full video processing pipeline.

    Args:
        video_path: Path to input video (accepts any format: MOV, MP4, M4A, etc.)
        output_dir: Directory for output files
        remove_pauses: Whether to remove long pauses
        pause_threshold: Silence duration in seconds to consider a pause
        pause_buffer: Buffer time in seconds to leave before/after cuts
        add_captions: Whether to add captions
        caption_style: Caption style ("tiktok", "mrbeast", "professional", "default")
        caption_mode: Caption mode ("word_by_word" or "segment")
        words_per_caption: Number of words per caption (used in segment mode)
        quality: Video quality preset ("fast", "balanced", "quality", "maximum")
        extract_audio: Whether to extract and keep M4A audio file
        normalize: Whether to normalize video to MP4 at the start
        clean_transcript: Whether to use AI to fix grammar in transcript
        content_edit: Whether to use AI to identify and remove repetitions/mistakes
    """
    # Setup
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    video_name = video_path.stem
    print("="*70)
    print("🎬 CELEVID - AI VIDEO EDITOR")
    print("="*70)
    print(f"Input: {video_path}")
    print(f"Output directory: {output_dir}")
    print(f"Quality: {quality}")
    print("="*70)

    # Step 0: Normalize video to MP4 (if enabled)
    current_video = str(video_path)
    if normalize:
        print("\n🔄 Normalizing video to MP4...")
        normalized_path = output_dir / f"{video_name}_normalized.mp4"
        normalize_video_to_mp4(str(video_path), str(normalized_path), quality)
        current_video = str(normalized_path)
        print(f"✓ Video normalized: {normalized_path}")

    # Step 1: Remove pauses first (if enabled)
    if remove_pauses:
        print("\n⏸️  Analyzing pauses...")
        editor = VideoEditor(current_video, pause_threshold, pause_buffer)
        pauses = editor.detect_silence()

        if pauses:
            print(f"Found {len(pauses)} long pauses")
            for i, (start, end, duration) in enumerate(pauses[:5], 1):
                print(f"  {i}. {start:.2f}s - {end:.2f}s ({duration:.1f}s)")

            # Apply edits
            edited_path = output_dir / f"{video_name}_edited.mp4"
            print(f"\n✂️  Removing pauses...")
            editor.apply_edits(str(edited_path))
            current_video = str(edited_path)
        else:
            print("✓ No long pauses detected")

    # Step 2: Extract audio to M4A (from edited video for accurate transcription)
    m4a_audio_path = None
    if extract_audio and AudioProcessor.has_audio(current_video):
        print("\n🎵 Extracting audio to M4A...")
        m4a_audio_path = output_dir / f"{video_name}_audio.m4a"
        AudioProcessor.extract_audio_to_m4a(current_video, str(m4a_audio_path))
        print(f"✓ Audio extracted: {m4a_audio_path}")

    # Step 3: Transcribe (AFTER pause removal for accurate timestamps)
    transcript = transcribe_video(current_video, output_dir, str(m4a_audio_path) if m4a_audio_path else None)

    # Step 4: Content editing - identify and remove repetitions/mistakes (optional)
    if content_edit:
        print("\n✂️  Analyzing content for cuts (repetitions, mistakes)...")
        try:
            content_editor = ContentEditorV2()
            cuts, segments = content_editor.analyze_transcript(transcript)

            if cuts:
                print(f"Found {len(cuts)} content cuts:")
                for start, end, reason in cuts:
                    print(f"  Cut {start:.2f}s - {end:.2f}s: {reason}")

                # Apply content cuts to video
                content_edited_path = output_dir / f"{video_name}_content_edited.mp4"
                editor = VideoEditor(current_video)
                editor.apply_content_cuts(
                    [(start, end) for start, end, _ in cuts],
                    str(content_edited_path)
                )
                current_video = str(content_edited_path)

                # Re-extract audio from content-edited video
                if extract_audio and AudioProcessor.has_audio(current_video):
                    print("\n🎵 Re-extracting audio from content-edited video...")
                    m4a_audio_path = output_dir / f"{video_name}_audio_final.m4a"
                    AudioProcessor.extract_audio_to_m4a(current_video, str(m4a_audio_path))

                # Re-transcribe the content-edited video
                print("\n📝 Re-transcribing content-edited video...")
                transcript = transcribe_video(current_video, output_dir, str(m4a_audio_path) if m4a_audio_path else None)
            else:
                print("✓ No content cuts needed")

        except Exception as e:
            print(f"⚠️  Content editing failed: {e}")
            print("  Continuing without content edits...")

    # Step 5: Clean transcript with AI (optional)
    if clean_transcript:
        print("\n🔍 Cleaning transcript with Gemini Flash...")
        try:
            cleaner = TranscriptCleaner()
            transcript = cleaner.clean_transcript(transcript)

            # Save cleaned transcript
            cleaned_transcript_path = output_dir / f"{video_name}_transcript_cleaned.json"
            with open(cleaned_transcript_path, 'w') as f:
                json.dump(transcript, f, indent=2)
            print(f"✓ Cleaned transcript saved: {cleaned_transcript_path}")
        except Exception as e:
            print(f"⚠️  Transcript cleaning failed: {e}")
            print("  Continuing with original transcript...")

    # Step 5: Add captions with quality encoding (optional)
    if add_captions:
        mode_display = "word-by-word" if caption_mode == "word_by_word" else "segment"
        print(f"\n🎨 Adding {caption_style} style captions ({mode_display} mode, quality: {quality})...")

        # Generate SRT file
        caption_service = CaptionService()
        srt_path = caption_service._generate_srt(transcript, words_per_caption, caption_mode)

        # Use quality encoding when burning in captions
        final_path = output_dir / f"{video_name}_final.mp4"

        try:
            # Build subtitles filter
            final_srt = output_dir / "temp_subtitles.srt"
            Path(srt_path).rename(final_srt)

            # Encode with quality settings
            process_video_with_quality(
                input_path=current_video,
                output_path=str(final_path),
                quality=quality,
                video_filter=f"subtitles={str(final_srt)}",
                copy_audio=True
            )

            print(f"✓ Captions added with {quality} quality")
            current_video = str(final_path)

        finally:
            # Cleanup temp SRT
            final_srt.unlink(missing_ok=True)

    # Summary
    print("\n" + "="*70)
    print("✅ PROCESSING COMPLETE")
    print("="*70)
    print(f"📹 Final video: {current_video}")
    print(f"📄 Transcript: {output_dir / f'{video_name}_transcript.json'}")
    if m4a_audio_path:
        print(f"🎵 Audio (M4A): {m4a_audio_path}")

    # Stats
    original_size = video_path.stat().st_size / 1024 / 1024
    final_size = Path(current_video).stat().st_size / 1024 / 1024
    print(f"📊 Size: {original_size:.1f}MB → {final_size:.1f}MB")

    if remove_pauses and pauses:
        total_pause_time = sum(d for _, _, d in pauses)
        print(f"⏱️  Time saved: {total_pause_time:.1f}s")

    return current_video


def process_multiple_videos(
    video_paths: list,
    output_dir: str = "output",
    output_name: str = "combined",
    remove_pauses: bool = True,
    pause_threshold: float = 1.0,
    pause_buffer: float = 0.5,
    add_captions: bool = True,
    caption_style: str = "tiktok",
    caption_mode: str = "word_by_word",
    words_per_caption: int = 3,
    quality: str = "balanced",
    extract_audio: bool = True,
    normalize: bool = True,
    clean_transcript: bool = True,
    content_edit: bool = True
):
    """
    Process multiple videos and combine them into one.

    Args:
        video_paths: List of input video paths
        output_dir: Directory for output files
        output_name: Name for the combined output video
        remove_pauses: Whether to remove long pauses
        pause_threshold: Silence duration in seconds to consider a pause
        pause_buffer: Buffer time in seconds to leave before/after cuts
        add_captions: Whether to add captions
        caption_style: Caption style ("tiktok", "mrbeast", "professional", "default")
        caption_mode: Caption mode ("word_by_word" or "segment")
        words_per_caption: Number of words per caption (used in segment mode)
        quality: Video quality preset ("fast", "balanced", "quality", "maximum")
        extract_audio: Whether to extract and keep M4A audio file
        normalize: Whether to normalize video to MP4 at the start
        clean_transcript: Whether to use AI to fix grammar in transcript
        content_edit: Whether to use AI to identify and remove repetitions/mistakes
    """
    import subprocess

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    print("="*70)
    print("🎬 CELEVID - BATCH VIDEO PROCESSOR")
    print("="*70)
    print(f"Processing {len(video_paths)} videos")
    print(f"Output directory: {output_dir}")
    print(f"Quality: {quality}")
    print("="*70)

    # Process each video individually
    processed_videos = []
    all_transcripts = []

    for i, video_path in enumerate(video_paths, 1):
        print(f"\n{'='*70}")
        print(f"📹 VIDEO {i}/{len(video_paths)}: {Path(video_path).name}")
        print(f"{'='*70}")

        # Process this video (clean transcript on combined, not individual)
        processed_path = process_video(
            video_path=video_path,
            output_dir=str(output_dir),
            remove_pauses=remove_pauses,
            pause_threshold=pause_threshold,
            pause_buffer=pause_buffer,
            add_captions=False,  # We'll add captions to the combined video
            quality=quality,
            extract_audio=extract_audio,
            normalize=normalize,
            clean_transcript=False,  # Clean merged transcript instead
            content_edit=content_edit  # Edit individual videos for repetitions
        )

        processed_videos.append(processed_path)

        # Load transcript for merging
        # The transcript filename is based on the processed video name
        processed_name = Path(processed_path).stem
        transcript_path = output_dir / f"{processed_name}_transcript.json"
        if transcript_path.exists():
            with open(transcript_path, 'r') as f:
                all_transcripts.append(json.load(f))
        else:
            # Try original video name as fallback
            video_name = Path(video_path).stem
            transcript_path = output_dir / f"{video_name}_transcript.json"
            if transcript_path.exists():
                with open(transcript_path, 'r') as f:
                    all_transcripts.append(json.load(f))

    # Concatenate all processed videos
    print(f"\n{'='*70}")
    print("🔗 COMBINING VIDEOS")
    print(f"{'='*70}")

    concat_file = output_dir / "concat_list.txt"
    with open(concat_file, 'w') as f:
        for video in processed_videos:
            f.write(f"file '{Path(video).absolute()}'\n")

    # Concatenate with re-encoding to fix audio/video sync issues
    concatenated_path = output_dir / f"{output_name}_concatenated.mp4"
    print(f"Concatenating {len(processed_videos)} videos with re-encoding...")

    # Build ffmpeg command with quality encoding
    from audio_processor import VideoQualityPresets

    cmd = [
        'ffmpeg',
        '-f', 'concat',
        '-safe', '0',
        '-i', str(concat_file),
    ]

    # Add video quality encoding params
    cmd.extend(VideoQualityPresets.get_encoding_params(quality))

    # Add audio encoding
    cmd.extend([
        '-c:a', 'aac',
        '-b:a', '192k',
        '-y',
        str(concatenated_path)
    ])

    subprocess.run(cmd, check=True, capture_output=True, text=True)

    print(f"✓ Videos concatenated: {concatenated_path}")

    # Merge transcripts with adjusted timestamps
    print("\n📝 Merging transcripts...")
    merged_transcript = merge_transcripts(all_transcripts, processed_videos)

    merged_transcript_path = output_dir / f"{output_name}_transcript.json"
    with open(merged_transcript_path, 'w') as f:
        json.dump(merged_transcript, f, indent=2)

    print(f"✓ Transcripts merged: {merged_transcript_path}")

    # Content editing on combined video - identify and remove repetitions/mistakes
    current_combined_video = str(concatenated_path)
    if content_edit:
        print("\n✂️  Analyzing combined video for content cuts (repetitions, mistakes)...")
        try:
            content_editor = ContentEditorV2()
            cuts, segments = content_editor.analyze_transcript(merged_transcript)

            if cuts:
                print(f"Found {len(cuts)} content cuts in combined video:")
                for start, end, reason in cuts:
                    print(f"  Cut {start:.2f}s - {end:.2f}s: {reason}")

                # Apply content cuts to combined video
                content_edited_path = output_dir / f"{output_name}_content_edited.mp4"
                editor = VideoEditor(current_combined_video)
                editor.apply_content_cuts(
                    [(start, end) for start, end, _ in cuts],
                    str(content_edited_path)
                )
                current_combined_video = str(content_edited_path)

                # Re-extract audio from content-edited combined video
                print("\n🎵 Re-extracting audio from content-edited combined video...")
                combined_audio_path = output_dir / f"{output_name}_audio.m4a"
                AudioProcessor.extract_audio_to_m4a(current_combined_video, str(combined_audio_path))

                # Re-transcribe the content-edited combined video
                print("\n📝 Re-transcribing content-edited combined video...")
                merged_transcript = transcribe_video(
                    current_combined_video,
                    output_dir,
                    str(combined_audio_path)
                )

                # Save updated transcript
                with open(merged_transcript_path, 'w') as f:
                    json.dump(merged_transcript, f, indent=2)
                print(f"✓ Updated transcript saved: {merged_transcript_path}")
            else:
                print("✓ No content cuts needed in combined video")

        except Exception as e:
            print(f"⚠️  Content editing failed: {e}")
            print("  Continuing without content edits...")

    # Clean merged transcript with AI (optional)
    if clean_transcript:
        print("\n🔍 Cleaning merged transcript with Gemini Flash...")
        try:
            cleaner = TranscriptCleaner()
            merged_transcript = cleaner.clean_transcript(merged_transcript)

            # Save cleaned transcript
            cleaned_transcript_path = output_dir / f"{output_name}_transcript_cleaned.json"
            with open(cleaned_transcript_path, 'w') as f:
                json.dump(merged_transcript, f, indent=2)
            print(f"✓ Cleaned transcript saved: {cleaned_transcript_path}")
        except Exception as e:
            print(f"⚠️  Transcript cleaning failed: {e}")
            print("  Continuing with original transcript...")

    # Add captions to the combined video (content-edited if cuts were made)
    final_path = Path(current_combined_video)
    if add_captions:
        mode_display = "word-by-word" if caption_mode == "word_by_word" else "segment"
        print(f"\n🎨 Adding {caption_style} style captions ({mode_display} mode) to combined video...")

        caption_service = CaptionService()
        srt_path = caption_service._generate_srt(merged_transcript, words_per_caption, caption_mode)

        final_path = output_dir / f"{output_name}_final.mp4"

        try:
            final_srt = output_dir / "temp_subtitles.srt"
            Path(srt_path).rename(final_srt)

            from audio_processor import process_video_with_quality
            process_video_with_quality(
                input_path=current_combined_video,
                output_path=str(final_path),
                quality=quality,
                video_filter=f"subtitles={str(final_srt)}",
                copy_audio=True
            )

            print(f"✓ Captions added with {quality} quality")
        finally:
            final_srt.unlink(missing_ok=True)

    # Cleanup concat file
    concat_file.unlink(missing_ok=True)

    # Summary
    print("\n" + "="*70)
    print("✅ BATCH PROCESSING COMPLETE")
    print("="*70)
    print(f"📹 Final combined video: {final_path}")
    print(f"📄 Combined transcript: {merged_transcript_path}")
    print(f"🎞️  Total clips combined: {len(video_paths)}")

    # Calculate total size
    total_original_size = sum(Path(v).stat().st_size for v in video_paths) / 1024 / 1024
    final_size = Path(final_path).stat().st_size / 1024 / 1024
    print(f"📊 Size: {total_original_size:.1f}MB → {final_size:.1f}MB")

    return str(final_path)


def merge_transcripts(transcripts: list, video_paths: list) -> dict:
    """
    Merge multiple transcripts with adjusted timestamps.

    Args:
        transcripts: List of transcript dicts
        video_paths: List of processed video paths (to get durations)

    Returns:
        Merged transcript dict
    """
    import subprocess

    merged = {
        "text": "",
        "chunks": []
    }

    current_offset = 0.0

    for i, (transcript, video_path) in enumerate(zip(transcripts, video_paths)):
        # Get video duration
        probe_cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'json',
            video_path
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True)
        duration = float(json.loads(result.stdout)['format']['duration'])

        # Add text
        if i > 0:
            merged["text"] += " "
        merged["text"] += transcript.get("text", "")

        # Add chunks with adjusted timestamps
        for chunk in transcript.get("chunks", []):
            adjusted_chunk = chunk.copy()
            if chunk["timestamp"][0] is not None:
                adjusted_chunk["timestamp"] = (
                    chunk["timestamp"][0] + current_offset,
                    chunk["timestamp"][1] + current_offset if chunk["timestamp"][1] else None
                )
            merged["chunks"].append(adjusted_chunk)

        # Update offset for next video
        current_offset += duration

    return merged


def main():
    """CLI interface for video processing."""
    if len(sys.argv) < 2:
        print("Usage: python process_video.py <video_file(s)> [options]")
        print("\nSingle video:")
        print("  python process_video.py video.mp4")
        print("\nMultiple videos (will be combined):")
        print("  python process_video.py video1.mp4 video2.mp4 video3.mp4")
        print("\nOptions:")
        print("  --no-pauses          Skip pause removal")
        print("  --no-captions        Skip caption generation")
        print("  --style STYLE        Caption style (tiktok, mrbeast, professional, default)")
        print("  --caption-mode MODE  Caption mode: word_by_word, segment (default: word_by_word)")
        print("  --threshold SEC      Pause threshold in seconds (default: 1.0)")
        print("  --buffer SEC         Buffer before/after cuts in seconds (default: 0.5)")
        print("  --words N            Words per caption for segment mode (default: 3)")
        print("  --output DIR         Output directory (default: output)")
        print("  --output-name NAME   Name for combined video (default: combined)")
        print("  --quality PRESET     Video quality: fast, balanced, quality, maximum (default: balanced)")
        print("  --no-audio           Skip M4A audio extraction")
        print("  --no-normalize       Skip video normalization to MP4")
        print("  --no-clean           Skip AI transcript cleaning (Gemini Flash)")
        print("  --no-content-edit    Skip AI content editing (removes repetitions/mistakes)")
        print("\nExamples:")
        print("  python process_video.py video.mp4")
        print("  python process_video.py video.mp4 --style mrbeast --words 4")
        print("  python process_video.py clip1.mp4 clip2.mp4 clip3.mp4 --output-name my_video")
        print("  python process_video.py video.mp4 --quality maximum")
        sys.exit(1)

    # Collect video paths (all non-flag arguments)
    video_paths = []
    args = []
    for arg in sys.argv[1:]:
        if arg.startswith('--'):
            args.extend(sys.argv[sys.argv.index(arg):])
            break
        else:
            video_paths.append(arg)

    # Parse options
    options = {
        'remove_pauses': '--no-pauses' not in args,
        'add_captions': '--no-captions' not in args,
        'caption_style': 'tiktok',
        'caption_mode': 'word_by_word',
        'pause_threshold': 1.0,
        'pause_buffer': 0.5,
        'words_per_caption': 3,
        'output_dir': 'output',
        'output_name': 'combined',
        'quality': 'balanced',
        'extract_audio': '--no-audio' not in args,
        'normalize': '--no-normalize' not in args,
        'clean_transcript': '--no-clean' not in args,
        'content_edit': '--no-content-edit' not in args
    }

    # Parse style
    if '--style' in args:
        idx = args.index('--style')
        if idx + 1 < len(args):
            options['caption_style'] = args[idx + 1]

    # Parse caption mode
    if '--caption-mode' in args:
        idx = args.index('--caption-mode')
        if idx + 1 < len(args):
            mode_value = args[idx + 1]
            if mode_value in ['word_by_word', 'segment']:
                options['caption_mode'] = mode_value
            else:
                print(f"Warning: Invalid caption mode '{mode_value}', using 'word_by_word'")

    # Parse threshold
    if '--threshold' in args:
        idx = args.index('--threshold')
        if idx + 1 < len(args):
            options['pause_threshold'] = float(args[idx + 1])

    # Parse buffer
    if '--buffer' in args:
        idx = args.index('--buffer')
        if idx + 1 < len(args):
            options['pause_buffer'] = float(args[idx + 1])

    # Parse words per caption
    if '--words' in args:
        idx = args.index('--words')
        if idx + 1 < len(args):
            options['words_per_caption'] = int(args[idx + 1])

    # Parse output dir
    if '--output' in args:
        idx = args.index('--output')
        if idx + 1 < len(args):
            options['output_dir'] = args[idx + 1]

    # Parse output name (for combined videos)
    if '--output-name' in args:
        idx = args.index('--output-name')
        if idx + 1 < len(args):
            options['output_name'] = args[idx + 1]

    # Parse quality
    if '--quality' in args:
        idx = args.index('--quality')
        if idx + 1 < len(args):
            quality_value = args[idx + 1]
            if quality_value in ['fast', 'balanced', 'quality', 'maximum']:
                options['quality'] = quality_value
            else:
                print(f"Warning: Invalid quality '{quality_value}', using 'balanced'")
                options['quality'] = 'balanced'

    # Process video(s)
    if len(video_paths) == 1:
        # Single video processing - remove output_name (only for batch)
        single_options = {k: v for k, v in options.items() if k != 'output_name'}
        process_video(video_paths[0], **single_options)
    else:
        # Multiple video processing (batch + combine)
        process_multiple_videos(video_paths, **options)


if __name__ == "__main__":
    main()
