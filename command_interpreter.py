"""
Natural language command parser for video editing operations.
Uses Gemini to parse user commands into structured actions.
"""

try:
    # Try new google.genai package first
    import google.genai as genai
    USE_NEW_API = True
except ImportError:
    # Fall back to deprecated google.generativeai
    import google.generativeai as genai
    USE_NEW_API = False

import json
import re
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path


class CommandInterpreter:
    """Parses natural language commands into structured video editing actions."""

    def __init__(self, api_key: str):
        """Initialize the command interpreter with Gemini API key."""
        if USE_NEW_API:
            self.client = genai.Client(api_key=api_key)
            self.model_name = 'gemini-2.0-flash-exp'
        else:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash-exp')

    def parse_command(self, user_input: str, context: dict) -> dict:
        """
        Parse natural language command into structured action.

        Args:
            user_input: User's natural language command
            context: Video context including:
                - duration: Total video duration in seconds
                - num_clips: Number of clips
                - transcript_snippet: Recent transcript words for context
                - clip_boundaries: List of (start, end) tuples for each clip

        Returns:
            Dictionary with:
                - action: "cut"|"move_clip"|"quality_check"|"add_captions"|"undo"|"help"|"done"
                - parameters: Action-specific parameters
                - confidence: Float 0-1
                - clarification_needed: Bool
                - clarification_question: Optional str
        """
        # Handle simple commands first
        user_input_lower = user_input.lower().strip()

        if user_input_lower in ['done', 'exit', 'quit', 'finish']:
            return {
                "action": "done",
                "parameters": {},
                "confidence": 1.0,
                "clarification_needed": False
            }

        if user_input_lower in ['help', '?']:
            return {
                "action": "help",
                "parameters": {},
                "confidence": 1.0,
                "clarification_needed": False
            }

        if user_input_lower in ['undo', 'revert', 'go back']:
            return {
                "action": "undo",
                "parameters": {},
                "confidence": 1.0,
                "clarification_needed": False
            }

        # Use Gemini for complex command parsing
        prompt = self._build_parsing_prompt(user_input, context)

        try:
            if USE_NEW_API:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt
                )
                result = json.loads(response.text.strip())
            else:
                response = self.model.generate_content(prompt)
                result = json.loads(response.text.strip())

            # Validate and normalize the result
            return self._validate_result(result, context)

        except json.JSONDecodeError as e:
            # Fallback if JSON parsing fails
            return {
                "action": "unknown",
                "parameters": {},
                "confidence": 0.0,
                "clarification_needed": True,
                "clarification_question": f"I couldn't understand that command. Could you rephrase it?"
            }
        except Exception as e:
            return {
                "action": "error",
                "parameters": {"error": str(e)},
                "confidence": 0.0,
                "clarification_needed": True,
                "clarification_question": f"An error occurred: {str(e)}"
            }

    def _build_parsing_prompt(self, user_input: str, context: dict) -> str:
        """Build the Gemini prompt for command parsing."""
        duration = context.get('duration', 0)
        num_clips = context.get('num_clips', 0)
        transcript_snippet = context.get('transcript_snippet', '')
        clip_boundaries = context.get('clip_boundaries', [])

        prompt = f"""You are a video editing command parser. Parse the user's command into a structured JSON action.

Context:
- Video duration: {duration:.1f} seconds ({self._format_time(duration)})
- Number of clips: {num_clips}
- Recent transcript: "{transcript_snippet}"
- Clip boundaries: {clip_boundaries}

User command: "{user_input}"

Available actions:
1. "cut" - Remove a time range from video
   Parameters: {{"start_time": float, "end_time": float}} OR {{"description": str}}
   Examples: "Remove from 1:23 to 2:45", "Cut the first 10 seconds", "Remove where I say um"

2. "move_clip" - Reorder clips
   Parameters: {{"clip_index": int, "target_position": int}}
   Examples: "Move clip 3 to the front", "Put last clip first"

3. "quality_check" - Run AI quality analysis
   Parameters: {{}}
   Examples: "Run quality check", "Find repetitions", "Check for mistakes"

4. "add_captions" - Add captions with style
   Parameters: {{"style": "tiktok"|"mrbeast"|"default"}}
   Examples: "Add MrBeast style captions", "Add captions", "Burn captions in TikTok style"

Respond ONLY with valid JSON in this exact format:
{{
  "action": "cut"|"move_clip"|"quality_check"|"add_captions"|"unknown",
  "parameters": {{}},
  "confidence": 0.0-1.0,
  "clarification_needed": true|false,
  "clarification_question": "optional question if clarification needed"
}}

Rules:
- For time-based cuts, parse times into seconds (float)
- For semantic cuts ("remove ums"), use description parameter
- Clip indices are 0-based
- Set confidence < 0.8 if uncertain
- Set clarification_needed=true if you need more info
- Always return valid JSON, no markdown formatting
"""
        return prompt

    def _validate_result(self, result: dict, context: dict) -> dict:
        """Validate and normalize the parsed result."""
        # Ensure required fields exist
        if 'action' not in result:
            result['action'] = 'unknown'
        if 'parameters' not in result:
            result['parameters'] = {}
        if 'confidence' not in result:
            result['confidence'] = 0.5
        if 'clarification_needed' not in result:
            result['clarification_needed'] = False

        # Validate action-specific parameters
        action = result['action']
        params = result['parameters']

        if action == 'cut':
            # Validate cut parameters
            if 'start_time' in params and 'end_time' in params:
                # Ensure times are within bounds
                duration = context.get('duration', 0)
                if params['end_time'] > duration:
                    result['clarification_needed'] = True
                    result['clarification_question'] = f"End time {params['end_time']:.1f}s exceeds video duration {duration:.1f}s"
                if params['start_time'] < 0:
                    params['start_time'] = 0
                if params['start_time'] >= params['end_time']:
                    result['clarification_needed'] = True
                    result['clarification_question'] = "Start time must be before end time"

        elif action == 'move_clip':
            # Validate clip indices
            num_clips = context.get('num_clips', 0)
            if 'clip_index' in params:
                if params['clip_index'] < 0 or params['clip_index'] >= num_clips:
                    result['clarification_needed'] = True
                    result['clarification_question'] = f"Clip index {params['clip_index']} out of range (0-{num_clips-1})"
            if 'target_position' in params:
                if params['target_position'] < 0 or params['target_position'] >= num_clips:
                    result['clarification_needed'] = True
                    result['clarification_question'] = f"Target position {params['target_position']} out of range (0-{num_clips-1})"

        elif action == 'add_captions':
            # Normalize style
            if 'style' not in params:
                params['style'] = 'default'
            if params['style'] not in ['tiktok', 'mrbeast', 'default']:
                params['style'] = 'default'

        return result

    def parse_timestamp(self, timestamp_str: str) -> Optional[float]:
        """
        Parse various timestamp formats into seconds.

        Supports:
        - "1:23" -> 83.0
        - "83s" -> 83.0
        - "1 minute 23 seconds" -> 83.0
        - "1m23s" -> 83.0
        """
        timestamp_str = timestamp_str.strip().lower()

        # Try MM:SS or HH:MM:SS format
        time_pattern = r'(\d+):(\d+)(?::(\d+))?'
        match = re.match(time_pattern, timestamp_str)
        if match:
            hours = int(match.group(1)) if match.group(3) else 0
            minutes = int(match.group(1)) if not match.group(3) else int(match.group(2))
            seconds = int(match.group(2)) if not match.group(3) else int(match.group(3))

            if match.group(3):  # HH:MM:SS
                return hours * 3600 + minutes * 60 + seconds
            else:  # MM:SS
                return minutes * 60 + seconds

        # Try "Xs" format
        if timestamp_str.endswith('s'):
            try:
                return float(timestamp_str[:-1])
            except ValueError:
                pass

        # Try plain number
        try:
            return float(timestamp_str)
        except ValueError:
            pass

        # Try "X minutes Y seconds" format
        minutes_match = re.search(r'(\d+)\s*(?:minute|min)', timestamp_str)
        seconds_match = re.search(r'(\d+)\s*(?:second|sec)', timestamp_str)

        if minutes_match or seconds_match:
            total = 0.0
            if minutes_match:
                total += int(minutes_match.group(1)) * 60
            if seconds_match:
                total += int(seconds_match.group(1))
            return total

        return None

    def _format_time(self, seconds: float) -> str:
        """Format seconds as MM:SS."""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}:{secs:02d}"

    def search_transcript_semantically(self, description: str, transcript: dict) -> List[Tuple[float, float, str]]:
        """
        Search transcript for segments matching description using Gemini.

        Args:
            description: Semantic description like "where I say um" or "repetitions"
            transcript: Transcript dict (can be segments or chunks format)

        Returns:
            List of (start_time, end_time, text) tuples for matching segments
        """
        # Handle different transcript formats
        if isinstance(transcript, dict):
            chunks = transcript.get('chunks', [])
            if not chunks:
                return []
        elif isinstance(transcript, list):
            chunks = transcript
        else:
            return []

        # Build transcript text with timestamps
        lines = []
        for i, chunk in enumerate(chunks):
            if isinstance(chunk, dict):
                text = chunk.get('text', '')
                timestamp = chunk.get('timestamp', [None, None])
                if timestamp and timestamp[0] is not None:
                    start = timestamp[0]
                    end = timestamp[1] if timestamp[1] is not None else start + 0.1
                    lines.append(f"[{i}] ({start:.1f}s-{end:.1f}s): {text}")

        transcript_text = "\n".join(lines)

        prompt = f"""Find segments in the transcript matching: "{description}"

Transcript:
{transcript_text}

Return a JSON array of matching segments with this format:
[
  {{"start_time": float, "end_time": float, "text": "matched text"}},
  ...
]

Rules:
- For "um", "uh", "like" - find filler words
- For "repetitions" - find repeated phrases
- For "mistakes" - find self-corrections
- Return empty array [] if no matches
- Always return valid JSON, no markdown formatting
"""

        try:
            if USE_NEW_API:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt
                )
                segments = json.loads(response.text.strip())
            else:
                response = self.model.generate_content(prompt)
                segments = json.loads(response.text.strip())

            # Convert to tuples
            return [(s['start_time'], s['end_time'], s['text']) for s in segments]

        except Exception as e:
            print(f"Error searching transcript: {e}")
            return []
