"""
Session state management for agentic video editing.
Tracks editing history, clip order, and enables undo functionality.
"""

import json
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime


class EditingSession:
    """Manages state across video editing iterations."""

    def __init__(self, video_paths: List[str], output_dir: str):
        """
        Initialize editing session.

        Args:
            video_paths: List of original video file paths
            output_dir: Directory for output files and state
        """
        self.video_paths = video_paths
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # State tracking
        self.clip_order = list(range(len(video_paths)))  # Initial order [0,1,2,...]
        self.current_video: Optional[str] = None
        self.current_transcript: Dict[str, Any] = {}
        self.edit_history: List[Dict[str, Any]] = []
        self.clip_boundaries: List[Tuple[float, float]] = []
        self.edit_counter = 0

        # Session metadata
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.state_file = self.output_dir / "session_state.json"

    def add_edit(self, edit_type: str, parameters: Dict[str, Any], video_path: str):
        """
        Record an edit operation for undo functionality.

        Args:
            edit_type: Type of edit ("initial", "cut", "move_clip", "quality_check", "add_captions")
            parameters: Edit-specific parameters
            video_path: Path to the resulting video file
        """
        edit_record = {
            "timestamp": datetime.now().isoformat(),
            "type": edit_type,
            "parameters": parameters,
            "video": video_path,
            "clip_order": self.clip_order.copy(),
            "clip_boundaries": self.clip_boundaries.copy()
        }

        self.edit_history.append(edit_record)
        self.current_video = video_path
        self.save_state()

    def undo_last_edit(self) -> Optional[Dict[str, Any]]:
        """
        Revert to previous edit state.

        Returns:
            Previous edit record, or None if no history
        """
        if len(self.edit_history) <= 1:
            # Can't undo initial processing
            return None

        # Remove current edit
        self.edit_history.pop()

        # Restore previous state
        previous_edit = self.edit_history[-1]
        self.current_video = previous_edit["video"]
        self.clip_order = previous_edit["clip_order"].copy()
        self.clip_boundaries = previous_edit["clip_boundaries"].copy()

        self.save_state()
        return previous_edit

    def reorder_clips(self, new_order: List[int]):
        """
        Update clip order.

        Args:
            new_order: New ordering of clip indices (e.g., [2,0,1])
        """
        if len(new_order) != len(self.video_paths):
            raise ValueError(f"New order must have {len(self.video_paths)} elements")

        if sorted(new_order) != list(range(len(self.video_paths))):
            raise ValueError("New order must be a permutation of clip indices")

        self.clip_order = new_order

    def get_next_edit_path(self) -> str:
        """
        Get path for next edit iteration.

        Returns:
            Path like "output/edit_5.mp4"
        """
        path = str(self.output_dir / f"edit_{self.edit_counter}.mp4")
        self.edit_counter += 1
        return path

    def get_ordered_video_paths(self) -> List[str]:
        """
        Get video paths in current clip order.

        Returns:
            List of video paths ordered according to clip_order
        """
        return [self.video_paths[i] for i in self.clip_order]

    def update_clip_boundaries(self, boundaries: List[Tuple[float, float]]):
        """
        Update clip boundary timestamps.

        Args:
            boundaries: List of (start, end) tuples for each clip in combined video
        """
        self.clip_boundaries = boundaries

    def update_transcript(self, transcript: Dict[str, Any]):
        """
        Update current transcript.

        Args:
            transcript: Transcript dictionary with segments
        """
        self.current_transcript = transcript

    def get_video_duration(self) -> float:
        """
        Get total duration of current video from clip boundaries.

        Returns:
            Duration in seconds
        """
        if not self.clip_boundaries:
            return 0.0
        return self.clip_boundaries[-1][1]

    def get_transcript_snippet(self, max_words: int = 20) -> str:
        """
        Get recent words from transcript for context.

        Args:
            max_words: Maximum number of recent words to include

        Returns:
            String of recent transcript words
        """
        if not self.current_transcript:
            return ""

        # Handle both 'segments' and 'chunks' formats
        segments = self.current_transcript.get('segments') or self.current_transcript.get('chunks', [])
        if not segments:
            # Try getting full text
            return self.current_transcript.get('text', '')[:100]

        # Get last few segments/chunks
        recent_segments = segments[-5:]
        words = []

        for seg in recent_segments:
            if isinstance(seg, dict):
                # Try both 'text' (segments) and other formats
                text = seg.get('text', '')
                words.extend(text.split())
            elif isinstance(seg, (list, tuple)):
                # Handle tuple format
                if len(seg) > 0 and isinstance(seg[0], str):
                    words.append(seg[0])

        # Return last max_words
        recent_words = words[-max_words:]
        return " ".join(recent_words)

    def get_context_dict(self) -> Dict[str, Any]:
        """
        Get context dictionary for command parsing.

        Returns:
            Dictionary with duration, num_clips, transcript_snippet, clip_boundaries
        """
        return {
            "duration": self.get_video_duration(),
            "num_clips": len(self.video_paths),
            "transcript_snippet": self.get_transcript_snippet(),
            "clip_boundaries": self.clip_boundaries
        }

    def save_state(self):
        """Persist session state to JSON file."""
        state = {
            "session_id": self.session_id,
            "video_paths": self.video_paths,
            "clip_order": self.clip_order,
            "current_video": self.current_video,
            "current_transcript": self.current_transcript,
            "edit_history": self.edit_history,
            "clip_boundaries": self.clip_boundaries,
            "edit_counter": self.edit_counter,
            "last_saved": datetime.now().isoformat()
        }

        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load_state(cls, output_dir: str) -> Optional['EditingSession']:
        """
        Restore session from saved state.

        Args:
            output_dir: Directory containing session_state.json

        Returns:
            Restored EditingSession, or None if no saved state
        """
        state_file = Path(output_dir) / "session_state.json"

        if not state_file.exists():
            return None

        try:
            with open(state_file, 'r') as f:
                state = json.load(f)

            # Create session instance
            session = cls(state["video_paths"], output_dir)

            # Restore state
            session.session_id = state["session_id"]
            session.clip_order = state["clip_order"]
            session.current_video = state["current_video"]
            session.current_transcript = state["current_transcript"]
            session.edit_history = state["edit_history"]
            session.clip_boundaries = state["clip_boundaries"]
            session.edit_counter = state["edit_counter"]

            return session

        except Exception as e:
            print(f"Error loading session state: {e}")
            return None

    def export_final_video(self, destination: Optional[str] = None) -> str:
        """
        Export current video as final output.

        Args:
            destination: Optional custom destination path

        Returns:
            Path to final video
        """
        if not self.current_video:
            raise ValueError("No video to export")

        if destination is None:
            destination = str(self.output_dir / "final_video.mp4")

        # Copy current video to final destination
        import shutil
        shutil.copy(self.current_video, destination)

        return destination

    def get_summary(self) -> str:
        """
        Get human-readable summary of current session state.

        Returns:
            Multi-line summary string
        """
        duration = self.get_video_duration()
        num_edits = len(self.edit_history) - 1  # Exclude initial processing

        summary_lines = [
            f"Session: {self.session_id}",
            f"Clips: {len(self.video_paths)} clips",
            f"Duration: {duration:.1f}s ({self._format_time(duration)})",
            f"Edits applied: {num_edits}",
            f"Current video: {self.current_video}"
        ]

        # Add clip order if changed
        if self.clip_order != list(range(len(self.video_paths))):
            summary_lines.append(f"Clip order: {self.clip_order}")

        return "\n".join(summary_lines)

    def _format_time(self, seconds: float) -> str:
        """Format seconds as MM:SS."""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}:{secs:02d}"
