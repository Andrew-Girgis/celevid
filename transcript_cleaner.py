"""
Transcript Quality Checker using Gemini Flash
Cleans up transcription errors while preserving word-level timestamps
"""
import os
import json
import time
from typing import Dict, List, Tuple
from dotenv import load_dotenv

load_dotenv()

from google import genai
from google.genai import types


class TranscriptCleaner:
    """Clean and improve transcripts using Gemini Flash"""

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")

        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-2.0-flash"

    def clean_transcript(self, transcript: Dict) -> Dict:
        """
        Clean transcript text while preserving timestamps.

        Args:
            transcript: Dict with 'text' and 'chunks' (word-level timestamps)

        Returns:
            Cleaned transcript with preserved/adjusted timestamps
        """
        chunks = transcript.get("chunks", [])
        if not chunks:
            return transcript

        # Extract original words with their timestamps
        original_words = []
        for chunk in chunks:
            text = chunk.get("text", "").strip()
            if text:
                original_words.append({
                    "word": text,
                    "start": chunk["timestamp"][0],
                    "end": chunk["timestamp"][1]
                })

        # Get the original text
        original_text = " ".join(w["word"] for w in original_words)

        # Ask Gemini to correct the text
        corrected_text = self._correct_with_gemini(original_text)

        if not corrected_text or corrected_text == original_text:
            print("  No corrections needed")
            return transcript

        print(f"  Original:  {original_text}")
        print(f"  Corrected: {corrected_text}")

        # Map corrected words back to timestamps
        corrected_chunks = self._map_corrections_to_timestamps(
            original_words,
            corrected_text
        )

        return {
            "text": corrected_text,
            "chunks": corrected_chunks
        }

    def _correct_with_gemini(self, text: str, max_retries: int = 3) -> str:
        """
        Use Gemini Flash to correct grammar and make text readable.
        Includes retry logic with exponential backoff for rate limits.
        """
        prompt = f"""Fix any grammar or transcription errors in this text. Make it read naturally.

Rules:
- Fix grammar (verb tenses, plurals, articles)
- Fix obvious transcription errors
- Keep the same meaning and words where possible
- Do NOT add explanations or quotes
- Do NOT change proper nouns or numbers
- Return ONLY the corrected text, nothing else

Text: {text}

Corrected:"""

        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.1,  # Low temperature for consistency
                        max_output_tokens=500,
                    )
                )

                corrected = response.text.strip()
                # Remove any quotes that Gemini might add
                corrected = corrected.strip('"\'')
                return corrected

            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    wait_time = (2 ** attempt) * 10  # 10s, 20s, 40s
                    print(f"  Rate limited. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                    time.sleep(wait_time)
                else:
                    print(f"  Gemini API error: {e}")
                    return text  # Return original on non-rate-limit errors

        print(f"  Max retries exceeded. Using original transcript.")
        return text

    def _map_corrections_to_timestamps(
        self,
        original_words: List[Dict],
        corrected_text: str
    ) -> List[Dict]:
        """
        Map corrected words back to original timestamps.

        Strategy:
        - Use sequence alignment to match words
        - Replacements keep original timestamp
        - Insertions share timestamp with adjacent word
        - Deletions are removed
        """
        corrected_words = corrected_text.split()

        # Simple word-by-word alignment using dynamic programming (simplified)
        aligned = self._align_words(
            [w["word"].lower().strip(".,!?") for w in original_words],
            [w.lower().strip(".,!?") for w in corrected_words]
        )

        result_chunks = []
        orig_idx = 0
        corr_idx = 0

        for op, orig_word, corr_word in aligned:
            if op == "match" or op == "replace":
                # Use original timestamp
                if orig_idx < len(original_words):
                    result_chunks.append({
                        "text": " " + corrected_words[corr_idx],
                        "timestamp": [
                            original_words[orig_idx]["start"],
                            original_words[orig_idx]["end"]
                        ]
                    })
                orig_idx += 1
                corr_idx += 1

            elif op == "insert":
                # New word - share timestamp with previous or next word
                if result_chunks:
                    # Share with previous word's end time
                    prev_end = result_chunks[-1]["timestamp"][1]
                    # Give it a small duration (0.15s)
                    result_chunks.append({
                        "text": " " + corrected_words[corr_idx],
                        "timestamp": [prev_end, prev_end + 0.15]
                    })
                elif orig_idx < len(original_words):
                    # Share with next word's start time
                    next_start = original_words[orig_idx]["start"]
                    result_chunks.append({
                        "text": " " + corrected_words[corr_idx],
                        "timestamp": [max(0, next_start - 0.15), next_start]
                    })
                corr_idx += 1

            elif op == "delete":
                # Word removed from transcript - skip it
                orig_idx += 1

        return result_chunks

    def _align_words(
        self,
        original: List[str],
        corrected: List[str]
    ) -> List[Tuple[str, str, str]]:
        """
        Align two word sequences using Levenshtein-style alignment.
        Returns list of (operation, original_word, corrected_word) tuples.
        """
        m, n = len(original), len(corrected)

        # DP table for edit distance
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        # Initialize
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if original[i-1] == corrected[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],    # delete
                        dp[i][j-1],    # insert
                        dp[i-1][j-1]   # replace
                    )

        # Backtrack to get alignment
        alignment = []
        i, j = m, n

        while i > 0 or j > 0:
            if i > 0 and j > 0 and original[i-1] == corrected[j-1]:
                alignment.append(("match", original[i-1], corrected[j-1]))
                i -= 1
                j -= 1
            elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
                alignment.append(("replace", original[i-1], corrected[j-1]))
                i -= 1
                j -= 1
            elif j > 0 and dp[i][j] == dp[i][j-1] + 1:
                alignment.append(("insert", None, corrected[j-1]))
                j -= 1
            else:
                alignment.append(("delete", original[i-1], None))
                i -= 1

        alignment.reverse()
        return alignment


def clean_transcript(transcript: Dict) -> Dict:
    """
    Convenience function to clean a transcript.

    Args:
        transcript: Dict with 'text' and 'chunks'

    Returns:
        Cleaned transcript
    """
    cleaner = TranscriptCleaner()
    return cleaner.clean_transcript(transcript)


if __name__ == "__main__":
    # Test with sample transcript
    test_transcript = {
        "text": "I built the tool that help creator make $300 million",
        "chunks": [
            {"text": " I", "timestamp": [0.32, 0.46]},
            {"text": " built", "timestamp": [0.46, 0.64]},
            {"text": " the", "timestamp": [0.64, 0.84]},
            {"text": " tool", "timestamp": [0.84, 1.08]},
            {"text": " that", "timestamp": [1.08, 1.48]},
            {"text": " help", "timestamp": [1.48, 1.72]},
            {"text": " creator", "timestamp": [1.72, 2.06]},
            {"text": " make", "timestamp": [2.06, 2.48]},
            {"text": " $300", "timestamp": [2.48, 2.98]},
            {"text": " million", "timestamp": [2.98, 3.42]},
        ]
    }

    print("Testing Transcript Cleaner with Gemini Flash...")
    print("=" * 50)

    cleaned = clean_transcript(test_transcript)

    print("\nResult:")
    print(f"Text: {cleaned['text']}")
    print("\nChunks:")
    for chunk in cleaned["chunks"]:
        print(f"  [{chunk['timestamp'][0]:.2f} - {chunk['timestamp'][1]:.2f}]: {chunk['text']}")
