"""
Script-based video orchestration using sentence-level segmentation.
Uses Gemini to intelligently segment, assess, and reorder content.
"""

import os
import json
from typing import List, Dict, Any, Tuple, Optional

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None
try:
    import google.genai as genai
    USE_NEW_API = True
except ImportError:
    import google.generativeai as genai
    USE_NEW_API = False


class SentenceSegment:
    """Represents a single sentence segment with quality metrics."""

    def __init__(
        self,
        text: str,
        start_time: float,
        end_time: float,
        clip_index: int,
        quality_score: float = 0.0,
        issues: List[str] = None
    ):
        self.text = text
        self.start_time = start_time
        self.end_time = end_time
        self.clip_index = clip_index
        self.quality_score = quality_score
        self.issues = issues or []
        self.original_index = None  # For tracking original order

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "text": self.text,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "clip_index": self.clip_index,
            "quality_score": self.quality_score,
            "issues": self.issues,
            "original_index": self.original_index
        }


class ScriptOrchestrator:
    """
    Orchestrates video editing at sentence level using AI.

    Workflow:
    1. Segment transcript into sentences
    2. Assess quality of each sentence
    3. Deduplicate and select best versions
    4. Reorder for optimal flow
    5. Generate edit instructions
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize with Gemini API."""
        # Load local .env (if available) so GEMINI_API_KEY can be provided without
        # manually exporting environment variables.
        if load_dotenv is not None:
            load_dotenv(override=False)

        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found")

        # Use the working google.genai API (not generativeai)
        self.client = genai.Client(api_key=api_key)
        # Use Gemini Exp 1206 - the most advanced experimental model
        self.model_name = 'gemini-exp-1206'

    def segment_into_sentences(self, transcript: dict) -> List[SentenceSegment]:
        """
        Segment transcript into complete sentences with timestamps.

        Args:
            transcript: Dict with 'chunks' (word-level timestamps)

        Returns:
            List of SentenceSegment objects
        """
        chunks = transcript.get('chunks', [])
        if not chunks:
            return []

        # Build word list with timestamps
        words = []
        for chunk in chunks:
            if isinstance(chunk, dict) and chunk.get('timestamp'):
                words.append({
                    'text': chunk.get('text', '').strip(),
                    'start': chunk['timestamp'][0],
                    'end': chunk['timestamp'][1] if chunk['timestamp'][1] else chunk['timestamp'][0] + 0.1
                })

        if not words:
            return []

        # Use Gemini to intelligently segment into sentences
        segments = self._segment_with_ai(words)

        return segments

    def _segment_with_ai(self, words: List[dict]) -> List[SentenceSegment]:
        """Use AI to segment words into logical sentences."""
        # Format words for AI
        word_text = " ".join([w['text'] for w in words])
        word_list = "\n".join([
            f"[{i}] ({w['start']:.2f}s-{w['end']:.2f}s): {w['text']}"
            for i, w in enumerate(words)
        ])

        prompt = f"""Segment these words into complete, logical sentences.

Full text: "{word_text}"

Words with timestamps:
{word_list}

Return a JSON array of sentences. For each sentence:
1. Identify start_word_index and end_word_index
2. Extract the complete sentence text
3. Note if sentence is incomplete or has issues

Format:
[
  {{
    "start_index": 0,
    "end_index": 15,
    "text": "Complete sentence here.",
    "is_complete": true,
    "issues": ["throat_clearing"] or []
  }}
]

Rules:
- Include EVERY word in a sentence
- Mark incomplete sentences (cut off, trailing words)
- Detect audio issues: throat clearing, stutters, false starts
- Combine fragments that belong together
- Return ONLY valid JSON, no markdown
"""

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )

            # Parse JSON response
            result_text = response.text.strip()
            # Remove markdown code blocks if present
            if result_text.startswith('```'):
                result_text = result_text.split('```')[1]
                if result_text.startswith('json'):
                    result_text = result_text[4:]

            sentence_data = json.loads(result_text.strip())

            # Convert to SentenceSegment objects
            segments = []
            for i, sent in enumerate(sentence_data):
                start_idx = sent['start_index']
                end_idx = sent['end_index']

                # Get timestamps from word indices
                start_time = words[start_idx]['start']
                end_time = words[end_idx]['end']

                issues = sent.get('issues', [])
                if not sent.get('is_complete', True):
                    issues.append('incomplete')

                segment = SentenceSegment(
                    text=sent['text'],
                    start_time=start_time,
                    end_time=end_time,
                    clip_index=0,  # Will be set by caller
                    quality_score=0.0,  # Will be assessed later
                    issues=issues
                )
                segment.original_index = i
                segments.append(segment)

            return segments

        except Exception as e:
            print(f"Error segmenting with AI: {e}")
            # Fallback: simple period-based segmentation
            return self._simple_segmentation(words)

    def _simple_segmentation(self, words: List[dict]) -> List[SentenceSegment]:
        """Fallback: simple period-based segmentation."""
        segments = []
        current_words = []
        start_time = None

        for i, word in enumerate(words):
            if start_time is None:
                start_time = word['start']

            current_words.append(word['text'])

            # End of sentence
            if word['text'].endswith(('.', '!', '?')) or i == len(words) - 1:
                text = " ".join(current_words).strip()
                segment = SentenceSegment(
                    text=text,
                    start_time=start_time,
                    end_time=word['end'],
                    clip_index=0,
                    quality_score=0.0
                )
                segments.append(segment)
                current_words = []
                start_time = None

        return segments

    def assess_quality(self, segments: List[SentenceSegment]) -> List[SentenceSegment]:
        """
        Assess quality of each sentence segment.

        Scores based on:
        - Grammar correctness
        - Audio quality indicators
        - Completeness
        - Clarity
        """
        # Build assessment prompt
        sentences_text = "\n".join([
            f"[{i}] \"{seg.text}\" (issues: {seg.issues})"
            for i, seg in enumerate(segments)
        ])

        prompt = f"""Assess the quality of each sentence for video editing.

Sentences:
{sentences_text}

For each sentence, provide:
1. quality_score (0.0-1.0): Overall quality
   - 1.0 = Perfect grammar, clear, complete
   - 0.7-0.9 = Good, minor issues
   - 0.4-0.6 = Mediocre, noticeable issues
   - 0.0-0.3 = Poor, major issues

2. issues: List of specific problems
   - "grammar": Grammatical errors
   - "incomplete": Sentence cut off or trailing
   - "unclear": Hard to understand
   - "throat_clearing": Audio artifacts
   - "stutter": Stuttering or repetition within sentence
   - "low_quality": Poor audio quality

Return JSON array:
[
  {{
    "index": 0,
    "quality_score": 0.95,
    "issues": [],
    "notes": "Perfect take"
  }},
  ...
]

Be critical - prefer grammatically correct, clear sentences.
Return ONLY valid JSON, no markdown.
"""

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )

            result_text = response.text.strip()
            if result_text.startswith('```'):
                result_text = result_text.split('```')[1]
                if result_text.startswith('json'):
                    result_text = result_text[4:]

            assessments = json.loads(result_text.strip())

            # Update segments with quality scores
            for assessment in assessments:
                idx = assessment['index']
                if 0 <= idx < len(segments):
                    segments[idx].quality_score = assessment['quality_score']
                    # Merge issues
                    new_issues = assessment.get('issues', [])
                    segments[idx].issues = list(set(segments[idx].issues + new_issues))

            return segments

        except Exception as e:
            print(f"Error assessing quality: {e}")
            # Fallback: basic scoring based on issues
            for seg in segments:
                score = 1.0
                if 'incomplete' in seg.issues:
                    score -= 0.3
                if 'throat_clearing' in seg.issues:
                    score -= 0.2
                if len(seg.issues) > 2:
                    score -= 0.2
                seg.quality_score = max(0.0, score)

            return segments

    def deduplicate_and_select(self, all_segments: List[SentenceSegment]) -> List[SentenceSegment]:
        """
        Find duplicate/similar sentences across clips and select best version.

        Args:
            all_segments: All segments from all clips

        Returns:
            Deduplicated list with best versions
        """
        # Use AI to find duplicates and select best
        segments_text = "\n".join([
            f"[{i}] (clip {seg.clip_index}, quality: {seg.quality_score:.2f}, issues: {seg.issues}): \"{seg.text}\""
            for i, seg in enumerate(all_segments)
        ])

        prompt = f"""Identify duplicate or similar sentences and select the best version of each.

Sentences:
{segments_text}

For each group of duplicates/similar sentences:
1. Group them together
2. Select the BEST version based on:
   - Highest quality_score
   - Fewest issues
   - Best grammar
   - Clearest audio

Return JSON:
{{
  "groups": [
    {{
      "sentence_indices": [0, 5, 12],
      "best_index": 5,
      "reason": "Better grammar and no throat clearing"
    }}
  ],
  "unique_indices": [1, 2, 3, 4, 6, 7, ...],
  "discard_indices": [0, 12]
}}

- groups: Duplicates with best selection
- unique_indices: Sentences with no duplicates
- discard_indices: Sentences to remove (worse duplicates)

Return ONLY valid JSON, no markdown.
"""

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )

            result_text = response.text.strip()
            if result_text.startswith('```'):
                result_text = result_text.split('```')[1]
                if result_text.startswith('json'):
                    result_text = result_text[4:]

            dedup_result = json.loads(result_text.strip())

            # Build final list of segments to keep
            keep_indices = set(dedup_result.get('unique_indices', []))
            for group in dedup_result.get('groups', []):
                keep_indices.add(group['best_index'])

            # Sort to maintain order
            keep_indices = sorted(keep_indices)
            selected_segments = [all_segments[i] for i in keep_indices]

            print(f"  Deduplication: {len(all_segments)} → {len(selected_segments)} segments")
            if dedup_result.get('groups'):
                print(f"  Found {len(dedup_result['groups'])} duplicate groups")

            return selected_segments

        except Exception as e:
            print(f"Error deduplicating: {e}")
            # Fallback: keep all segments
            return all_segments

    def orchestrate_script(self, segments: List[SentenceSegment]) -> List[SentenceSegment]:
        """
        Reorder segments for optimal narrative flow.

        Args:
            segments: Deduplicated segments

        Returns:
            Reordered segments
        """
        segments_text = "\n".join([
            f"[{i}] \"{seg.text}\""
            for i, seg in enumerate(segments)
        ])

        prompt = f"""Reorder these sentences for the best narrative flow and coherence.

Current order:
{segments_text}

Create a logical, engaging script by:
1. Starting with introduction/hook
2. Building logical flow
3. Ending with conclusion/call-to-action
4. Removing truly unnecessary repetitions

Return JSON with new order:
{{
  "new_order": [2, 0, 1, 5, 3, 4],  // Indices in new sequence
  "removed": [6, 7],  // Indices to remove completely
  "reasoning": "Started with hook, removed unnecessary repetition..."
}}

Return ONLY valid JSON, no markdown.
"""

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )

            result_text = response.text.strip()
            if result_text.startswith('```'):
                result_text = result_text.split('```')[1]
                if result_text.startswith('json'):
                    result_text = result_text[4:]

            orchestration = json.loads(result_text.strip())

            # Reorder segments
            new_order = orchestration.get('new_order', list(range(len(segments))))
            removed = set(orchestration.get('removed', []))

            reordered = [
                segments[i] for i in new_order
                if i not in removed and 0 <= i < len(segments)
            ]

            print(f"  Orchestration: {orchestration.get('reasoning', 'Reordered for better flow')}")
            if removed:
                print(f"  Removed {len(removed)} unnecessary segments")

            return reordered

        except Exception as e:
            print(f"Error orchestrating: {e}")
            # Fallback: keep original order
            return segments

    def generate_edit_plan(
        self,
        final_segments: List[SentenceSegment],
        clip_boundaries: List[Tuple[float, float]]
    ) -> Dict[str, Any]:
        """
        Generate video editing instructions based on orchestrated script.

        Args:
            final_segments: Final ordered segments to keep
            clip_boundaries: Timestamp boundaries for each clip

        Returns:
            Edit plan with cuts and concatenation order
        """
        edit_plan = {
            "segments_to_keep": [],
            "segments_to_remove": [],
            "final_script": " ".join([seg.text for seg in final_segments]),
            "total_duration": 0.0
        }

        for seg in final_segments:
            edit_plan["segments_to_keep"].append({
                "text": seg.text,
                "start": seg.start_time,
                "end": seg.end_time,
                "clip": seg.clip_index,
                "quality": seg.quality_score
            })
            edit_plan["total_duration"] += (seg.end_time - seg.start_time)

        return edit_plan

    def process_transcript(
        self,
        transcript: dict,
        clip_index: int = 0
    ) -> List[SentenceSegment]:
        """
        Full processing pipeline for a single transcript.

        Args:
            transcript: Transcript dict with chunks
            clip_index: Index of the clip this transcript belongs to

        Returns:
            Processed segments with quality scores
        """
        print(f"\n  📝 Segmenting into sentences...")
        segments = self.segment_into_sentences(transcript)
        print(f"     Found {len(segments)} sentences")

        # Set clip index
        for seg in segments:
            seg.clip_index = clip_index

        print(f"  🔍 Assessing quality...")
        segments = self.assess_quality(segments)

        # Show quality summary
        avg_quality = sum(s.quality_score for s in segments) / len(segments) if segments else 0
        print(f"     Average quality: {avg_quality:.2f}")
        low_quality = [s for s in segments if s.quality_score < 0.5]
        if low_quality:
            print(f"     ⚠️  {len(low_quality)} low-quality segments detected")

        return segments
