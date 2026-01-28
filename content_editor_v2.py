"""
Enhanced content editor using sentence-level orchestration.
Replaces word-level editing with intelligent script orchestration.
"""

from typing import List, Tuple, Dict, Any
from script_orchestrator import ScriptOrchestrator, SentenceSegment


class ContentEditorV2:
    """
    Enhanced content editor that works at sentence level.

    Improvements over V1:
    - Sentence-level understanding (not word-level)
    - Quality assessment (grammar, audio quality)
    - Smart deduplication (keeps best takes)
    - Script orchestration (logical flow)
    - Uses better model (Gemini 2.0 Flash Thinking)
    """

    def __init__(self):
        """Initialize with ScriptOrchestrator."""
        self.orchestrator = ScriptOrchestrator()

    def analyze_transcript(
        self,
        transcript: Dict,
        clip_index: int = 0
    ) -> Tuple[List[Tuple[float, float, str]], List[SentenceSegment]]:
        """
        Analyze transcript and generate cuts using sentence-level orchestration.

        Args:
            transcript: Dict with 'text' and 'chunks' (word-level timestamps)
            clip_index: Index of the clip (for multi-clip processing)

        Returns:
            Tuple of (cuts, segments)
            - cuts: List of (start_time, end_time, reason) tuples for segments to REMOVE
            - segments: List of SentenceSegment objects (for reference)
        """
        print(f"\n  🎬 Analyzing content with sentence-level orchestration...")

        # Process transcript into quality-assessed sentences
        segments = self.orchestrator.process_transcript(transcript, clip_index)

        if not segments:
            print("  ✓ No segments found")
            return [], []

        # Identify low-quality segments to cut
        cuts = self._generate_cuts(segments)

        return cuts, segments

    def analyze_multiple_transcripts(
        self,
        transcripts: List[Dict]
    ) -> Tuple[List[Tuple[float, float, str]], Dict[str, Any]]:
        """
        Analyze multiple transcripts (from different clips) together.

        This is the power of the new approach:
        1. Process each transcript into sentences
        2. Deduplicate across ALL clips
        3. Select best version of each sentence
        4. Orchestrate into optimal script
        5. Generate cuts to achieve final script

        Args:
            transcripts: List of transcript dicts

        Returns:
            Tuple of (cuts, orchestration_result)
        """
        print(f"\n🎭 SCRIPT ORCHESTRATION (Multiple Clips)")
        print("=" * 70)

        # Step 1: Process each transcript
        all_segments = []
        for i, transcript in enumerate(transcripts):
            print(f"\n📹 Clip {i + 1}:")
            segments = self.orchestrator.process_transcript(transcript, clip_index=i)
            all_segments.extend(segments)

        print(f"\n  Total sentences across all clips: {len(all_segments)}")

        # Step 2: Deduplicate and select best versions
        print(f"\n  🔄 Deduplicating and selecting best takes...")
        selected_segments = self.orchestrator.deduplicate_and_select(all_segments)

        # Step 3: Orchestrate for optimal flow
        print(f"\n  🎯 Orchestrating script for optimal flow...")
        final_segments = self.orchestrator.orchestrate_script(selected_segments)

        # Step 4: Generate cuts
        print(f"\n  ✂️  Generating edit instructions...")
        cuts = self._generate_cuts_from_orchestration(all_segments, final_segments)

        # Build result
        result = {
            "original_segments": [s.to_dict() for s in all_segments],
            "final_segments": [s.to_dict() for s in final_segments],
            "final_script": " ".join([s.text for s in final_segments]),
            "cuts": cuts,
            "stats": {
                "original_count": len(all_segments),
                "final_count": len(final_segments),
                "removed_count": len(all_segments) - len(final_segments)
            }
        }

        print(f"\n📊 Results:")
        print(f"  • Original: {len(all_segments)} sentences")
        print(f"  • Final: {len(final_segments)} sentences")
        print(f"  • Removed: {len(all_segments) - len(final_segments)} sentences")
        print(f"\n  Final script: \"{result['final_script']}\"")
        print("=" * 70)

        return cuts, result

    def _generate_cuts(self, segments: List[SentenceSegment]) -> List[Tuple[float, float, str]]:
        """
        Generate cuts from quality-assessed segments.

        Remove segments that are:
        - Very low quality (< 0.4)
        - Incomplete
        - Have multiple serious issues
        """
        cuts = []

        for seg in segments:
            should_cut = False
            reason_parts = []

            # Quality threshold
            if seg.quality_score < 0.4:
                should_cut = True
                reason_parts.append(f"low quality ({seg.quality_score:.2f})")

            # Specific issues
            if 'incomplete' in seg.issues:
                should_cut = True
                reason_parts.append("incomplete sentence")

            if 'grammar' in seg.issues and seg.quality_score < 0.6:
                should_cut = True
                reason_parts.append("grammatical errors")

            if len(seg.issues) >= 3:
                should_cut = True
                reason_parts.append(f"multiple issues: {', '.join(seg.issues)}")

            if should_cut:
                reason = "; ".join(reason_parts)
                cuts.append((seg.start_time, seg.end_time, reason))

        return cuts

    def _generate_cuts_from_orchestration(
        self,
        original_segments: List[SentenceSegment],
        final_segments: List[SentenceSegment]
    ) -> List[Tuple[float, float, str]]:
        """
        Generate cuts by comparing original and final segments.

        Any segment in original that's not in final should be cut.
        """
        # Create set of final segment timestamps for quick lookup
        final_timestamps = set([
            (s.start_time, s.end_time, s.clip_index)
            for s in final_segments
        ])

        cuts = []
        for seg in original_segments:
            key = (seg.start_time, seg.end_time, seg.clip_index)
            if key not in final_timestamps:
                # This segment was removed during orchestration
                reason = f"Removed during orchestration (quality: {seg.quality_score:.2f}"
                if seg.issues:
                    reason += f", issues: {', '.join(seg.issues)}"
                reason += ")"
                cuts.append((seg.start_time, seg.end_time, reason))

        return cuts

    def explain_decision(self, segment: SentenceSegment, kept: bool) -> str:
        """Generate human-readable explanation of why segment was kept/removed."""
        if kept:
            return f"✅ KEPT: \"{segment.text}\" (quality: {segment.quality_score:.2f})"
        else:
            issues_str = ", ".join(segment.issues) if segment.issues else "none"
            return f"❌ CUT: \"{segment.text}\" (quality: {segment.quality_score:.2f}, issues: {issues_str})"
