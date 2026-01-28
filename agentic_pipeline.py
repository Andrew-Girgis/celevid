"""
Agentic video editing pipeline with natural language command interface.
Orchestrates multi-clip video editing with iterative user commands.
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import subprocess

from command_interpreter import CommandInterpreter
from session_state import EditingSession
from process_video import process_multiple_videos, transcribe_video
from content_editor import ContentEditor
from video_editor import VideoEditor
from caption_service import CaptionService


class AgenticVideoEditor:
    """Main orchestrator for interactive multi-clip video editing."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the agentic editor."""
        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found in environment variables")

        self.api_key = api_key
        self.interpreter = CommandInterpreter(api_key)
        self.content_editor = ContentEditor()
        self.session: Optional[EditingSession] = None

    def start_session(
        self,
        video_paths: List[str],
        output_dir: str = "output",
        pause_threshold: float = 1.0,
        skip_initial_quality_check: bool = False
    ):
        """
        Start an interactive editing session.

        Args:
            video_paths: List of input video file paths
            output_dir: Directory for output files
            pause_threshold: Silence duration threshold for pause removal
            skip_initial_quality_check: Skip AI quality check during initial processing
        """
        # Validate inputs
        for path in video_paths:
            if not Path(path).exists():
                raise FileNotFoundError(f"Video file not found: {path}")

        # Create session
        self.session = EditingSession(video_paths, output_dir)

        print("=" * 70)
        print("🎬 AGENTIC VIDEO EDITOR")
        print("=" * 70)
        print(f"Clips: {len(video_paths)}")
        for i, path in enumerate(video_paths):
            print(f"  {i+1}. {Path(path).name}")
        print("=" * 70)

        # Initial processing
        print("\n=== INITIAL PROCESSING ===")
        initial_video = self._run_initial_processing(
            video_paths,
            output_dir,
            pause_threshold,
            skip_initial_quality_check
        )

        if not initial_video:
            print("❌ Initial processing failed")
            return

        # Load transcript and calculate clip boundaries
        self._load_transcript_and_boundaries(initial_video)

        # Record initial state
        self.session.add_edit(
            "initial",
            {
                "video_paths": video_paths,
                "pause_threshold": pause_threshold,
                "quality_check": not skip_initial_quality_check
            },
            initial_video
        )

        # Print summary
        print("\n" + "=" * 70)
        print("Summary:")
        duration = self.session.get_video_duration()
        print(f"- Combined duration: {duration:.1f}s ({self._format_time(duration)})")
        print(f"- Clips: {len(video_paths)} clips")
        if self.session.current_transcript.get('segments'):
            word_count = sum(len(seg.get('text', '').split()) for seg in self.session.current_transcript['segments'])
            print(f"- Transcript: {word_count} words")
        print("=" * 70)

        # Enter interactive loop
        self._interactive_loop()

    def _run_initial_processing(
        self,
        video_paths: List[str],
        output_dir: str,
        pause_threshold: float,
        skip_quality_check: bool
    ) -> Optional[str]:
        """Run initial video processing with stitching and quality checks."""
        print("1. Stitching clips together...")
        print(f"2. Removing pauses ({pause_threshold}s+ threshold)...")
        print("3. Transcribing video...")
        if not skip_quality_check:
            print("4. Running AI quality check...")

        try:
            # Use existing process_multiple_videos function
            process_multiple_videos(
                video_paths=video_paths,
                output_dir=output_dir,
                output_name="working_video",
                remove_pauses=True,
                pause_threshold=pause_threshold,
                add_captions=False,  # Don't add captions yet
                content_edit=not skip_quality_check,  # AI quality check
                extract_audio=True,
                normalize=True,
                clean_transcript=True
            )

            # Return path to processed video
            # Check for content_edited version first (if quality check was run)
            working_video_content = Path(output_dir) / "working_video_content_edited.mp4"
            working_video = Path(output_dir) / "working_video.mp4"

            if working_video_content.exists():
                print("✅ Initial processing complete!")
                return str(working_video_content)
            elif working_video.exists():
                print("✅ Initial processing complete!")
                return str(working_video)
            else:
                # Check for concatenated version
                working_video_concat = Path(output_dir) / "working_video_concatenated.mp4"
                if working_video_concat.exists():
                    print("✅ Initial processing complete!")
                    return str(working_video_concat)
                return None

        except Exception as e:
            print(f"Error during initial processing: {e}")
            return None

    def _load_transcript_and_boundaries(self, video_path: str):
        """Load transcript and calculate clip boundaries."""
        # Try video-specific transcript first
        transcript_path = Path(video_path).with_suffix('.json')

        # Also try the cleaned version
        video_stem = Path(video_path).stem
        transcript_cleaned = Path(video_path).parent / f"{video_stem}_transcript_cleaned.json"
        transcript_regular = Path(video_path).parent / f"{video_stem}_transcript.json"

        # Try different transcript files in order of preference
        for tpath in [transcript_cleaned, transcript_regular, transcript_path]:
            if tpath.exists():
                with open(tpath, 'r') as f:
                    transcript = json.load(f)
                    self.session.update_transcript(transcript)
                break

        # Calculate clip boundaries
        # For now, we'll need to infer from the combined video duration
        # In a full implementation, we'd track this during stitching
        duration = self._get_video_duration(video_path)

        # Simple approximation: divide duration equally among clips
        # (This is a simplification - real implementation would track during concat)
        num_clips = len(self.session.video_paths)
        clip_duration = duration / num_clips

        boundaries = []
        for i in range(num_clips):
            start = i * clip_duration
            end = (i + 1) * clip_duration
            boundaries.append((start, end))

        self.session.update_clip_boundaries(boundaries)

    def _get_video_duration(self, video_path: str) -> float:
        """Get video duration using ffprobe."""
        try:
            result = subprocess.run(
                [
                    'ffprobe',
                    '-v', 'error',
                    '-show_entries', 'format=duration',
                    '-of', 'default=noprint_wrappers=1:nokey=1',
                    video_path
                ],
                capture_output=True,
                text=True,
                check=True
            )
            return float(result.stdout.strip())
        except Exception as e:
            print(f"Error getting video duration: {e}")
            return 0.0

    def _interactive_loop(self):
        """Main interactive editing loop."""
        print("\n" + "=" * 70)
        print("=== AGENTIC EDITING MODE ===")
        print("=" * 70)
        print("Tell me what you'd like to edit. Examples:")
        print('  - "Remove from 1:23 to 2:45"')
        print('  - "Cut the part where I say um too much"')
        print('  - "Move clip 3 to the beginning"')
        print('  - "Add captions in MrBeast style"')
        print('  - "Run quality check"')
        print("\nType 'help' for more commands, 'done' to finish.")
        print("=" * 70)

        while True:
            try:
                # Get user input
                user_input = input("\n>> ").strip()

                if not user_input:
                    continue

                # Parse command
                context = self.session.get_context_dict()
                command = self.interpreter.parse_command(user_input, context)

                # Handle clarification
                if command.get('clarification_needed'):
                    print(f"❓ {command.get('clarification_question', 'Could you clarify?')}")
                    continue

                # Execute command
                success = self._execute_command(command)

                if not success:
                    print("❌ Command failed")

                # Check if done
                if command['action'] == 'done':
                    break

            except KeyboardInterrupt:
                print("\n\n⚠️  Interrupted. Type 'done' to finish or continue editing.")
                continue
            except Exception as e:
                print(f"❌ Error: {e}")
                continue

        # Export final video
        self._finalize_session()

    def _execute_command(self, command: dict) -> bool:
        """Route command to appropriate handler."""
        action = command['action']

        try:
            if action == 'cut':
                return self._handle_cut(command['parameters'])
            elif action == 'move_clip':
                return self._handle_move_clip(command['parameters'])
            elif action == 'quality_check':
                return self._handle_quality_check()
            elif action == 'add_captions':
                return self._handle_add_captions(command['parameters'])
            elif action == 'undo':
                return self._handle_undo()
            elif action == 'help':
                return self._handle_help()
            elif action == 'done':
                return True
            else:
                print(f"Unknown action: {action}")
                return False

        except Exception as e:
            print(f"Error executing {action}: {e}")
            return False

    def _handle_cut(self, params: dict) -> bool:
        """Handle cut command (time-based or semantic)."""
        if 'description' in params:
            # Semantic cut - search transcript
            return self._handle_semantic_cut(params['description'])
        elif 'start_time' in params and 'end_time' in params:
            # Time-based cut
            return self._handle_time_cut(params['start_time'], params['end_time'])
        else:
            print("Invalid cut parameters")
            return False

    def _handle_time_cut(self, start_time: float, end_time: float) -> bool:
        """Apply time-based cut."""
        print(f"✂️  Cutting from {self._format_time(start_time)} to {self._format_time(end_time)}...")

        try:
            # Get current video
            input_video = self.session.current_video
            output_video = self.session.get_next_edit_path()

            # Use VideoEditor to apply cut
            editor = VideoEditor(input_video)
            cuts = [(start_time, end_time)]

            # Apply cuts (remove segments)
            editor.apply_content_cuts(cuts, output_video)

            # Re-transcribe
            print("📝 Re-transcribing...")
            self._transcribe_video(output_video)

            # Update session
            self.session.add_edit(
                "cut",
                {"cuts": cuts},
                output_video
            )

            print("✅ Cut applied successfully!")
            return True

        except Exception as e:
            print(f"Error applying cut: {e}")
            return False

    def _handle_semantic_cut(self, description: str) -> bool:
        """Handle semantic cut based on transcript search."""
        print(f"🔍 Searching transcript for: {description}")

        # Search transcript
        if not self.session.current_transcript:
            print("No transcript available")
            return False

        segments = self.interpreter.search_transcript_semantically(
            description,
            self.session.current_transcript
        )

        if not segments:
            print("❌ No matching segments found")
            return False

        # Display found segments
        print(f"Found {len(segments)} segment(s):")
        for i, (start, end, text) in enumerate(segments, 1):
            print(f"  {i}. {self._format_time(start)} - {self._format_time(end)}: \"{text}\"")

        # Ask for confirmation
        response = input("Remove all these segments? [y/n]: ").strip().lower()
        if response != 'y':
            print("Cancelled")
            return False

        # Apply cuts
        print("✂️  Applying cuts...")
        try:
            input_video = self.session.current_video
            output_video = self.session.get_next_edit_path()

            editor = VideoEditor(input_video)
            cuts = [(start, end) for start, end, _ in segments]

            editor.apply_content_cuts(cuts, output_video)

            # Re-transcribe
            print("📝 Re-transcribing...")
            self._transcribe_video(output_video)

            # Update session
            self.session.add_edit(
                "cut",
                {"cuts": cuts, "description": description},
                output_video
            )

            print("✅ Cuts applied successfully!")
            return True

        except Exception as e:
            print(f"Error applying cuts: {e}")
            return False

    def _handle_move_clip(self, params: dict) -> bool:
        """Handle clip reordering."""
        clip_index = params.get('clip_index')
        target_position = params.get('target_position')

        if clip_index is None or target_position is None:
            print("Invalid move_clip parameters")
            return False

        print(f"📦 Moving clip {clip_index + 1} to position {target_position + 1}...")

        try:
            # Update clip order
            new_order = self.session.clip_order.copy()
            clip_to_move = new_order.pop(clip_index)
            new_order.insert(target_position, clip_to_move)

            self.session.reorder_clips(new_order)

            # Re-stitch videos in new order
            print("🔄 Re-stitching video...")
            output_video = self.session.get_next_edit_path()

            self._stitch_clips(self.session.get_ordered_video_paths(), output_video)

            # Re-transcribe
            print("📝 Re-transcribing...")
            self._transcribe_video(output_video)

            # Update session
            self.session.add_edit(
                "move_clip",
                {"from": clip_index, "to": target_position, "new_order": new_order},
                output_video
            )

            print("✅ Clips reordered successfully!")
            return True

        except Exception as e:
            print(f"Error moving clip: {e}")
            return False

    def _handle_quality_check(self) -> bool:
        """Run AI quality check and optionally apply fixes."""
        print("🔍 Running AI quality check...")

        try:
            # Analyze transcript with ContentEditor
            cuts = self.content_editor.analyze_transcript(self.session.current_transcript)

            if not cuts:
                print("✅ No issues found!")
                return True

            # Display found issues
            print(f"Found {len(cuts)} issue(s):")
            for i, (start, end, reason) in enumerate(cuts, 1):
                print(f"  {i}. {self._format_time(start)} - {self._format_time(end)}: {reason}")

            # Ask for confirmation
            response = input("Apply these cuts? [y/n]: ").strip().lower()
            if response != 'y':
                print("Cancelled")
                return False

            # Apply cuts
            print("✂️  Applying quality fixes...")
            input_video = self.session.current_video
            output_video = self.session.get_next_edit_path()

            editor = VideoEditor(input_video)
            editor.apply_content_cuts(cuts, output_video)

            # Re-transcribe
            print("📝 Re-transcribing...")
            self._transcribe_video(output_video)

            # Update session
            self.session.add_edit(
                "quality_check",
                {"cuts": cuts},
                output_video
            )

            print("✅ Quality fixes applied!")
            return True

        except Exception as e:
            print(f"Error running quality check: {e}")
            return False

    def _handle_add_captions(self, params: dict) -> bool:
        """Add captions with specified style."""
        style = params.get('style', 'default')
        print(f"🎨 Adding {style} captions...")

        try:
            input_video = self.session.current_video
            output_video = self.session.get_next_edit_path()

            # Use CaptionService
            caption_service = CaptionService()
            transcript = self.session.current_transcript

            # Add captions
            caption_service.add_captions(
                input_video,
                transcript,
                output_video,
                style=style,
                mode="word_by_word"
            )

            # Update session
            self.session.add_edit(
                "add_captions",
                {"style": style},
                output_video
            )

            print("✅ Captions added!")
            return True

        except Exception as e:
            print(f"Error adding captions: {e}")
            return False

    def _handle_undo(self) -> bool:
        """Undo last edit."""
        previous_edit = self.session.undo_last_edit()

        if previous_edit is None:
            print("❌ Nothing to undo (can't undo initial processing)")
            return False

        print(f"↶ Reverted to previous state")
        print(f"Current video: {self.session.current_video}")
        return True

    def _handle_help(self) -> bool:
        """Display help message."""
        print("\n" + "=" * 70)
        print("AVAILABLE COMMANDS")
        print("=" * 70)
        print("\nTime-based cuts:")
        print('  - "Remove from 1:23 to 2:45"')
        print('  - "Cut the first 10 seconds"')
        print('  - "Delete from 0:30 to 1:00"')
        print("\nSemantic cuts:")
        print('  - "Remove where I say um"')
        print('  - "Cut the repetitions"')
        print('  - "Delete the mistakes"')
        print("\nClip reordering:")
        print('  - "Move clip 3 to the front"')
        print('  - "Put last clip first"')
        print('  - "Reorder: clip 2 to position 0"')
        print("\nQuality operations:")
        print('  - "Run quality check"')
        print('  - "Find and remove repetitions"')
        print("\nCaptions:")
        print('  - "Add MrBeast style captions"')
        print('  - "Add captions in TikTok style"')
        print('  - "Burn captions"')
        print("\nOther:")
        print('  - "undo" - Revert last edit')
        print('  - "help" - Show this message')
        print('  - "done" - Finish and export final video')
        print("=" * 70)
        return True

    def _stitch_clips(self, video_paths: List[str], output_path: str):
        """Stitch multiple clips using FFmpeg concat."""
        # Create concat file
        concat_file = Path(output_path).parent / "concat_list.txt"

        with open(concat_file, 'w') as f:
            for path in video_paths:
                # Make path absolute for FFmpeg
                abs_path = Path(path).resolve()
                f.write(f"file '{abs_path}'\n")

        # Run FFmpeg concat
        cmd = [
            'ffmpeg',
            '-f', 'concat',
            '-safe', '0',
            '-i', str(concat_file),
            '-c', 'copy',
            '-y',
            output_path
        ]

        subprocess.run(cmd, check=True, capture_output=True)

        # Clean up
        concat_file.unlink()

    def _transcribe_video(self, video_path: str):
        """Transcribe video and update session."""
        try:
            # Use existing transcribe_video function
            output_dir = Path(video_path).parent
            transcript = transcribe_video(video_path, output_dir)

            # Save transcript
            transcript_path = Path(video_path).with_suffix('.json')
            with open(transcript_path, 'w') as f:
                json.dump(transcript, f, indent=2)

            # Update session
            self.session.update_transcript(transcript)

            # Update duration from transcript chunks
            if transcript.get('chunks'):
                # Find last chunk with valid timestamp
                last_chunk = None
                for chunk in reversed(transcript['chunks']):
                    if chunk.get('timestamp') and chunk['timestamp'][1] is not None:
                        last_chunk = chunk
                        break

                if last_chunk:
                    duration = last_chunk['timestamp'][1]
                    # Update clip boundaries proportionally
                    self._update_clip_boundaries_for_duration(duration)

        except Exception as e:
            print(f"Warning: Transcription failed: {e}")

    def _update_clip_boundaries_for_duration(self, new_duration: float):
        """Update clip boundaries after video duration changes."""
        # Simple proportional update
        # (Real implementation would track boundaries more carefully)
        old_duration = self.session.get_video_duration()
        if old_duration == 0:
            return

        scale = new_duration / old_duration
        new_boundaries = [
            (start * scale, end * scale)
            for start, end in self.session.clip_boundaries
        ]

        self.session.update_clip_boundaries(new_boundaries)

    def _finalize_session(self):
        """Export final video and show summary."""
        print("\n" + "=" * 70)
        print("💾 Exporting final video...")

        try:
            final_path = self.session.export_final_video()
            print(f"✅ Final video saved: {final_path}")

            # Show session summary
            print("\n" + self.session.get_summary())

            print("\nSession state saved to:", self.session.state_file)
            print("=" * 70)

        except Exception as e:
            print(f"❌ Error exporting final video: {e}")

    def _format_time(self, seconds: float) -> str:
        """Format seconds as MM:SS."""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}:{secs:02d}"


def main():
    """CLI entry point for agentic mode."""
    if len(sys.argv) < 2:
        print("Usage: python agentic_pipeline.py video1.mp4 video2.mp4 [video3.mp4 ...]")
        print("\nOptions:")
        print("  --output DIR          Output directory (default: output)")
        print("  --pause-threshold SEC Pause detection threshold (default: 1.0)")
        print("  --skip-quality-check  Skip initial AI quality check")
        sys.exit(1)

    # Parse arguments
    video_paths = []
    output_dir = "output"
    pause_threshold = 1.0
    skip_quality_check = False

    for arg in sys.argv[1:]:
        if arg.startswith('--'):
            if arg == '--skip-quality-check':
                skip_quality_check = True
            elif arg == '--output':
                # Next arg is the directory
                idx = sys.argv.index(arg)
                if idx + 1 < len(sys.argv):
                    output_dir = sys.argv[idx + 1]
            elif arg == '--pause-threshold':
                idx = sys.argv.index(arg)
                if idx + 1 < len(sys.argv):
                    pause_threshold = float(sys.argv[idx + 1])
        elif not arg.replace('.', '').replace('-', '').replace('_', '').isalnum():
            # Likely a file path
            if Path(arg).exists():
                video_paths.append(arg)

    if not video_paths:
        print("❌ No valid video files provided")
        sys.exit(1)

    # Create editor and start session
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
        sys.exit(1)


if __name__ == "__main__":
    main()
