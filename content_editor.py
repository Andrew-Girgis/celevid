"""
AI-Assisted Content Editor using Gemini Flash
Identifies segments to cut: repetitions, "cut that out", filler words, mistakes
"""
import os
import json
import time
import re
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv

load_dotenv()

from google import genai
from google.genai import types


class ContentEditor:
    """Analyze transcript and identify segments to cut from video"""

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")

        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-2.0-flash"

    def analyze_transcript(self, transcript: Dict) -> List[Tuple[float, float, str]]:
        """
        Analyze transcript and identify segments that should be cut.

        Args:
            transcript: Dict with 'text' and 'chunks' (word-level timestamps)

        Returns:
            List of (start_time, end_time, reason) tuples for segments to cut
        """
        # Handle different transcript formats
        if not isinstance(transcript, dict):
            print(f"  Warning: transcript is not a dict, got {type(transcript)}")
            return []

        chunks = transcript.get("chunks", [])
        if not chunks:
            return []

        # Build transcript with timestamps for analysis
        words_with_times = []
        for chunk in chunks:
            # Handle both dict and other formats
            if isinstance(chunk, dict):
                text = chunk.get("text", "").strip()
                timestamp = chunk.get("timestamp", [None, None])
            elif isinstance(chunk, (list, tuple)) and len(chunk) >= 2:
                # Handle tuple/list format: (text, (start, end)) or similar
                text = str(chunk[0]).strip() if chunk[0] else ""
                timestamp = chunk[1] if isinstance(chunk[1], (list, tuple)) else [None, None]
            else:
                continue

            if text and timestamp[0] is not None:
                start, end = timestamp
                words_with_times.append({
                    "word": text,
                    "start": start,
                    "end": end if end is not None else start + 0.1
                })

        # Format for Gemini
        transcript_text = self._format_transcript_for_analysis(words_with_times)

        # Ask Gemini to identify cuts
        cuts = self._identify_cuts_with_gemini(transcript_text, words_with_times)

        return cuts

    def _format_transcript_for_analysis(self, words_with_times: List[Dict]) -> str:
        """Format transcript with timestamps for Gemini analysis"""
        lines = []
        for i, w in enumerate(words_with_times):
            lines.append(f"[{i}] ({w['start']:.2f}s-{w['end']:.2f}s): {w['word']}")
        return "\n".join(lines)

    def _identify_cuts_with_gemini(
        self,
        transcript_text: str,
        words_with_times: List[Dict],
        max_retries: int = 3
    ) -> List[Tuple[float, float, str]]:
        """
        Use Gemini to identify segments that should be cut.
        """
        prompt = f"""Analyze this transcript and identify segments that should be CUT from the video.

Each line shows: [word_index] (start_time-end_time): word

Look for:
1. REPETITIONS: Where speaker restarts a sentence (e.g., "I built the- I built the tool")
2. SELF-CORRECTIONS: Phrases like "wait", "let me start over", "cut that", "actually no"
3. FILLER WORDS: Excessive "um", "uh", "like", "you know" (only if disruptive)
4. FALSE STARTS: Incomplete thoughts that are restarted
5. MISTAKES: Where speaker clearly messes up and corrects themselves

Return ONLY a JSON array of cuts. Each cut should have:
- "start_index": first word index to cut
- "end_index": last word index to cut (inclusive)
- "reason": brief explanation

If NO cuts are needed, return an empty array: []

IMPORTANT:
- Be conservative - only cut clear mistakes/repetitions
- Don't cut natural speech patterns
- Don't cut content that adds meaning

Transcript:
{transcript_text}

Return ONLY valid JSON array, nothing else:"""

        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        max_output_tokens=1000,
                    )
                )

                result_text = response.text.strip()

                # Clean up response - extract JSON array
                result_text = self._extract_json_array(result_text)

                # Parse JSON
                cuts_data = json.loads(result_text)

                if not isinstance(cuts_data, list):
                    print("  Gemini returned invalid format, no cuts identified")
                    return []

                # Convert to timestamp tuples
                cuts = []
                for cut in cuts_data:
                    # Handle both dict format and list format
                    if isinstance(cut, dict):
                        start_idx = cut.get("start_index", 0)
                        end_idx = cut.get("end_index", 0)
                        reason = cut.get("reason", "unknown")
                    elif isinstance(cut, (list, tuple)) and len(cut) >= 2:
                        # Handle list format: [start_index, end_index, reason]
                        start_idx = int(cut[0]) if cut[0] is not None else 0
                        end_idx = int(cut[1]) if cut[1] is not None else 0
                        reason = str(cut[2]) if len(cut) > 2 else "unknown"
                    else:
                        continue

                    if start_idx < len(words_with_times) and end_idx < len(words_with_times):
                        start_time = words_with_times[start_idx]["start"]
                        end_time = words_with_times[end_idx]["end"]
                        cuts.append((start_time, end_time, reason))

                return cuts

            except json.JSONDecodeError as e:
                print(f"  Failed to parse Gemini response as JSON: {e}")
                return []

            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    wait_time = (2 ** attempt) * 10
                    print(f"  Rate limited. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                    time.sleep(wait_time)
                else:
                    print(f"  Gemini API error: {e}")
                    return []

        print(f"  Max retries exceeded. No cuts identified.")
        return []

    def _extract_json_array(self, text: str) -> str:
        """Extract JSON array from potentially messy response"""
        # Remove markdown code blocks
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        text = text.strip()

        # Find the JSON array
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            return match.group(0)
        return "[]"


def analyze_for_cuts(transcript: Dict) -> List[Tuple[float, float, str]]:
    """
    Convenience function to analyze transcript for content cuts.

    Args:
        transcript: Dict with 'text' and 'chunks'

    Returns:
        List of (start_time, end_time, reason) tuples
    """
    editor = ContentEditor()
    return editor.analyze_transcript(transcript)


if __name__ == "__main__":
    # Test with sample transcript that has repetition
    test_transcript = {
        "text": "I built the wait let me start over I built the tool that helps creators",
        "chunks": [
            {"text": " I", "timestamp": [0.0, 0.2]},
            {"text": " built", "timestamp": [0.2, 0.4]},
            {"text": " the", "timestamp": [0.4, 0.6]},
            {"text": " wait", "timestamp": [0.6, 0.9]},
            {"text": " let", "timestamp": [0.9, 1.1]},
            {"text": " me", "timestamp": [1.1, 1.3]},
            {"text": " start", "timestamp": [1.3, 1.6]},
            {"text": " over", "timestamp": [1.6, 1.9]},
            {"text": " I", "timestamp": [1.9, 2.1]},
            {"text": " built", "timestamp": [2.1, 2.3]},
            {"text": " the", "timestamp": [2.3, 2.5]},
            {"text": " tool", "timestamp": [2.5, 2.8]},
            {"text": " that", "timestamp": [2.8, 3.0]},
            {"text": " helps", "timestamp": [3.0, 3.3]},
            {"text": " creators", "timestamp": [3.3, 3.7]},
        ]
    }

    print("Testing Content Editor with Gemini Flash...")
    print("=" * 50)
    print(f"Input: {test_transcript['text']}")
    print("=" * 50)

    cuts = analyze_for_cuts(test_transcript)

    print(f"\nIdentified {len(cuts)} cuts:")
    for start, end, reason in cuts:
        print(f"  Cut {start:.2f}s - {end:.2f}s: {reason}")
