import librosa
import numpy as np
import subprocess
import json
from pathlib import Path
from typing import List, Tuple
import sys


class VideoEditor:
    def __init__(self, video_path: str, pause_threshold: float = 1.5, buffer: float = 0.5):
        """
        Initialize video editor.

        Args:
            video_path: Path to input video
            pause_threshold: Silence duration in seconds to consider a "long pause"
            buffer: Buffer time in seconds to leave before/after cuts (default: 0.5s)
        """
        self.video_path = video_path
        self.pause_threshold = pause_threshold
        self.buffer = buffer
        self.silence_segments = []
        self.transcript_data = None

    def detect_silence(self) -> List[Tuple[float, float]]:
        """Detect silent segments in the audio."""
        print(f"Analyzing audio for pauses longer than {self.pause_threshold}s...")
        print(f"Using {self.buffer}s buffer before/after cuts")

        # Load audio
        audio, sr = librosa.load(self.video_path, sr=16000)

        # Detect non-silent intervals
        intervals = librosa.effects.split(audio, top_db=30)

        # Find gaps between speech (silence segments)
        silence_segments = []

        # Check for silence at the start (before first speech)
        if len(intervals) > 0:
            first_speech_start = intervals[0][0] / sr
            if first_speech_start >= self.pause_threshold:
                # Apply buffer: keep some silence before speech
                cut_start = 0.0
                cut_end = max(0.0, first_speech_start - self.buffer)
                cut_duration = cut_end - cut_start

                if cut_duration > 0:
                    silence_segments.append((cut_start, cut_end, cut_duration))

        # Check gaps between speech segments
        for i in range(len(intervals) - 1):
            silence_start = intervals[i][1] / sr
            silence_end = intervals[i + 1][0] / sr
            silence_duration = silence_end - silence_start

            if silence_duration >= self.pause_threshold:
                # Apply buffer: keep buffer on both sides
                cut_start = silence_start + self.buffer
                cut_end = silence_end - self.buffer
                cut_duration = cut_end - cut_start

                # Only add if there's still something to cut after buffer
                if cut_duration > 0:
                    silence_segments.append((cut_start, cut_end, cut_duration))

        # Check for silence at the end (after last speech)
        if len(intervals) > 0:
            audio_duration = len(audio) / sr
            last_speech_end = intervals[-1][1] / sr
            end_silence_duration = audio_duration - last_speech_end
            if end_silence_duration >= self.pause_threshold:
                # Apply buffer: keep some silence after speech
                cut_start = last_speech_end + self.buffer
                cut_end = audio_duration
                cut_duration = cut_end - cut_start

                if cut_duration > 0:
                    silence_segments.append((cut_start, cut_end, cut_duration))

        self.silence_segments = silence_segments
        print(f"Found {len(silence_segments)} long pauses")
        return silence_segments

    def create_annotated_preview(self, output_path: str):
        """Create a preview video with visual markers showing what will be cut."""
        if not self.silence_segments:
            print("No edits to preview")
            return

        print(f"Creating preview video with {len(self.silence_segments)} annotations...")

        # Build ffmpeg filter for text overlays
        drawtext_filters = []

        for i, (start, end, duration) in enumerate(self.silence_segments):
            # Add text overlay during the silence
            text = f"⚠ LONG PAUSE ({duration:.1f}s) - WILL BE CUT"
            drawtext = (
                f"drawtext="
                f"text='{text}':"
                f"fontsize=40:"
                f"fontcolor=white:"
                f"box=1:"
                f"boxcolor=red@0.8:"
                f"boxborderw=10:"
                f"x=(w-text_w)/2:"
                f"y=h-100:"
                f"enable='between(t,{start},{end})'"
            )
            drawtext_filters.append(drawtext)

        # Combine all drawtext filters
        filter_chain = ",".join(drawtext_filters)

        # Run ffmpeg to create annotated video
        cmd = [
            'ffmpeg', '-i', self.video_path,
            '-vf', filter_chain,
            '-c:a', 'copy',  # Copy audio without re-encoding
            '-y',  # Overwrite output
            output_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error creating preview: {result.stderr}")
            return False

        print(f"✓ Preview saved to: {output_path}")
        return True

    def apply_edits(self, output_path: str):
        """Create final edited video with long pauses removed."""
        if not self.silence_segments:
            print("No edits to apply")
            return

        print(f"Applying {len(self.silence_segments)} edits...")

        # Get video duration
        probe_cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'json',
            self.video_path
        ]
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
        duration = float(json.loads(probe_result.stdout)['format']['duration'])

        # Build list of segments to keep (inverse of silence segments)
        keep_segments = []
        current_pos = 0.0

        for silence_start, silence_end, _ in self.silence_segments:
            if silence_start > current_pos:
                keep_segments.append((current_pos, silence_start))
            current_pos = silence_end

        # Add final segment if needed
        if current_pos < duration:
            keep_segments.append((current_pos, duration))

        print(f"Keeping {len(keep_segments)} segments, removing {len(self.silence_segments)} pauses")

        # Create a concat file for ffmpeg
        concat_file = Path(output_path).parent / "concat_list.txt"
        temp_segments = []

        try:
            # Extract each segment to keep (with re-encoding to fix timestamps)
            for i, (start, end) in enumerate(keep_segments):
                segment_path = Path(output_path).parent / f"segment_{i}.mp4"
                temp_segments.append(segment_path)

                cmd = [
                    'ffmpeg',
                    '-ss', str(start),  # Seek before input for faster processing
                    '-i', self.video_path,
                    '-to', str(end - start),  # Duration from start
                    '-c:v', 'libx264',  # Re-encode video
                    '-preset', 'fast',  # Fast encoding preset
                    '-crf', '18',  # High quality
                    '-c:a', 'aac',  # Re-encode audio
                    '-b:a', '192k',  # Audio bitrate
                    '-avoid_negative_ts', 'make_zero',  # Fix timestamp issues
                    '-y',
                    str(segment_path)
                ]
                subprocess.run(cmd, capture_output=True, check=True)

            # Create concat file
            with open(concat_file, 'w') as f:
                for seg in temp_segments:
                    f.write(f"file '{seg.absolute()}'\n")

            # Concatenate all segments (with re-encoding for consistency)
            cmd = [
                'ffmpeg',
                '-f', 'concat',
                '-safe', '0',
                '-i', str(concat_file),
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '18',
                '-c:a', 'aac',
                '-b:a', '192k',
                '-y',
                output_path
            ]
            subprocess.run(cmd, capture_output=True, check=True)

            print(f"✓ Final edited video saved to: {output_path}")

            # Calculate time saved
            removed_time = sum(duration for _, _, duration in self.silence_segments)
            print(f"✓ Removed {removed_time:.1f}s of pauses")
            print(f"✓ Original: {duration:.1f}s → Final: {duration - removed_time:.1f}s")

        finally:
            # Cleanup temp files
            for seg in temp_segments:
                seg.unlink(missing_ok=True)
            concat_file.unlink(missing_ok=True)

        return True

    def apply_content_cuts(self, cuts: List[Tuple[float, float]], output_path: str):
        """
        Apply content-based cuts (not silence-based).

        Args:
            cuts: List of (start_time, end_time) tuples to remove
            output_path: Path for output video

        Returns:
            True if successful
        """
        if not cuts:
            print("No content cuts to apply")
            return False

        print(f"Applying {len(cuts)} content cuts...")

        # Get video duration
        probe_cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'json',
            self.video_path
        ]
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
        duration = float(json.loads(probe_result.stdout)['format']['duration'])

        # Sort cuts by start time and merge overlapping
        sorted_cuts = sorted(cuts, key=lambda x: x[0])
        merged_cuts = []
        for start, end in sorted_cuts:
            if merged_cuts and start <= merged_cuts[-1][1]:
                # Merge with previous cut
                merged_cuts[-1] = (merged_cuts[-1][0], max(merged_cuts[-1][1], end))
            else:
                merged_cuts.append((start, end))

        # Build list of segments to keep (inverse of cuts)
        keep_segments = []
        current_pos = 0.0

        for cut_start, cut_end in merged_cuts:
            if cut_start > current_pos:
                keep_segments.append((current_pos, cut_start))
            current_pos = cut_end

        # Add final segment if needed
        if current_pos < duration:
            keep_segments.append((current_pos, duration))

        print(f"Keeping {len(keep_segments)} segments, removing {len(merged_cuts)} cuts")

        # Create a concat file for ffmpeg
        concat_file = Path(output_path).parent / "content_concat_list.txt"
        temp_segments = []

        try:
            # Extract each segment to keep
            for i, (start, end) in enumerate(keep_segments):
                segment_path = Path(output_path).parent / f"content_segment_{i}.mp4"
                temp_segments.append(segment_path)

                cmd = [
                    'ffmpeg',
                    '-ss', str(start),
                    '-i', self.video_path,
                    '-to', str(end - start),
                    '-c:v', 'libx264',
                    '-preset', 'fast',
                    '-crf', '18',
                    '-c:a', 'aac',
                    '-b:a', '192k',
                    '-avoid_negative_ts', 'make_zero',
                    '-y',
                    str(segment_path)
                ]
                subprocess.run(cmd, capture_output=True, check=True)

            # Create concat file
            with open(concat_file, 'w') as f:
                for seg in temp_segments:
                    f.write(f"file '{seg.absolute()}'\n")

            # Concatenate all segments
            cmd = [
                'ffmpeg',
                '-f', 'concat',
                '-safe', '0',
                '-i', str(concat_file),
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '18',
                '-c:a', 'aac',
                '-b:a', '192k',
                '-y',
                output_path
            ]
            subprocess.run(cmd, capture_output=True, check=True)

            # Calculate time removed
            removed_time = sum(end - start for start, end in merged_cuts)
            print(f"✓ Content-edited video saved to: {output_path}")
            print(f"✓ Removed {removed_time:.1f}s of content")
            print(f"✓ Original: {duration:.1f}s -> Final: {duration - removed_time:.1f}s")

        finally:
            # Cleanup temp files
            for seg in temp_segments:
                seg.unlink(missing_ok=True)
            concat_file.unlink(missing_ok=True)

        return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python video_editor.py <video_file> [pause_threshold]")
        print("Example: python video_editor.py assets/Ricky_1.mov 1.5")
        sys.exit(1)

    video_path = sys.argv[1]
    pause_threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 1.5

    editor = VideoEditor(video_path, pause_threshold)

    # Detect long pauses
    silence_segments = editor.detect_silence()

    if not silence_segments:
        print("No long pauses found! Video is already tight.")
        return

    # Show what was found
    print("\nLong pauses detected:")
    for i, (start, end, duration) in enumerate(silence_segments, 1):
        print(f"  {i}. {start:.2f}s - {end:.2f}s ({duration:.1f}s pause)")

    # Create preview with annotations
    video_name = Path(video_path).stem
    preview_path = f"assets/{video_name}_preview.mp4"
    editor.create_annotated_preview(preview_path)

    # Ask user if they want to apply edits
    print("\n" + "="*60)
    print("Preview created! Review the annotated video.")
    response = input("Apply edits and create final video? [y/n]: ").strip().lower()

    if response == 'y':
        final_path = f"assets/{video_name}_edited.mp4"
        editor.apply_edits(final_path)
    else:
        print("Edits cancelled. Preview saved for review.")


if __name__ == "__main__":
    main()
