#!/usr/bin/env python3
"""
Test script to demonstrate sentence-level orchestration.
Shows the improvement over word-level editing.
"""

import json
import os
import sys
from pathlib import Path
from content_editor_v2 import ContentEditorV2


def load_transcript(json_path: str) -> dict:
    """Load transcript from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def test_single_clip():
    """Test orchestration on a single clip."""
    print("=" * 70)
    print("TEST 1: Single Clip Analysis")
    print("=" * 70)

    # Load Ricky 2 transcript (the one with grammar issues)
    transcript_path = "output/Ricky 2_edited_transcript.json"

    if not Path(transcript_path).exists():
        print(f"❌ Transcript not found: {transcript_path}")
        print("   Run the pipeline first to generate transcripts")
        return

    transcript = load_transcript(transcript_path)

    print(f"\nOriginal transcript:")
    print(f'"{transcript["text"]}"')

    # Analyze with V2
    editor = ContentEditorV2()
    cuts, segments = editor.analyze_transcript(transcript, clip_index=1)

    # Show results
    print(f"\n📊 Analysis Results:")
    print(f"  • Total sentences: {len(segments)}")
    print(f"  • Cuts recommended: {len(cuts)}")

    if segments:
        print(f"\n📝 Sentence Quality Breakdown:")
        for i, seg in enumerate(segments):
            issues_str = f" (issues: {', '.join(seg.issues)})" if seg.issues else ""
            print(f"  {i + 1}. [{seg.quality_score:.2f}] \"{seg.text}\"{issues_str}")

    if cuts:
        print(f"\n✂️  Recommended Cuts:")
        for start, end, reason in cuts:
            print(f"  • {start:.1f}s - {end:.1f}s: {reason}")


def test_multi_clip():
    """Test orchestration across multiple clips."""
    print("\n\n" + "=" * 70)
    print("TEST 2: Multi-Clip Orchestration")
    print("=" * 70)

    # Load all three Ricky transcripts
    transcript_paths = [
        "output/Ricky 1_edited_transcript.json",
        "output/Ricky 2_edited_transcript.json",
        "output/Ricky 3_edited_transcript.json"
    ]

    transcripts = []
    for path in transcript_paths:
        if not Path(path).exists():
            print(f"❌ Transcript not found: {path}")
            return
        transcripts.append(load_transcript(path))

    print(f"\nOriginal transcripts:")
    for i, t in enumerate(transcripts):
        print(f"\nClip {i + 1}: \"{t['text']}\"")

    # Analyze with V2
    editor = ContentEditorV2()
    cuts, result = editor.analyze_multiple_transcripts(transcripts)

    # Show detailed results
    print(f"\n\n📋 DETAILED ORCHESTRATION RESULTS:")
    print("=" * 70)

    print(f"\n🎬 Original Sentences (All Clips):")
    for i, seg_dict in enumerate(result['original_segments']):
        quality = seg_dict['quality_score']
        issues = seg_dict.get('issues', [])
        issues_str = f" ⚠️  {', '.join(issues)}" if issues else ""
        emoji = "✅" if quality >= 0.7 else "⚠️ " if quality >= 0.4 else "❌"
        print(f"  {i + 1}. {emoji} [Clip {seg_dict['clip_index'] + 1}] [{quality:.2f}] \"{seg_dict['text']}\"{issues_str}")

    print(f"\n🎯 Final Script (After Orchestration):")
    for i, seg_dict in enumerate(result['final_segments']):
        print(f"  {i + 1}. [Clip {seg_dict['clip_index'] + 1}] \"{seg_dict['text']}\"")

    print(f"\n✂️  What Got Cut:")
    removed_count = result['stats']['removed_count']
    if removed_count > 0:
        # Find removed segments
        final_texts = set(s['text'] for s in result['final_segments'])
        for seg_dict in result['original_segments']:
            if seg_dict['text'] not in final_texts:
                reason_parts = []
                if seg_dict['quality_score'] < 0.6:
                    reason_parts.append(f"low quality: {seg_dict['quality_score']:.2f}")
                if seg_dict.get('issues'):
                    reason_parts.append(f"issues: {', '.join(seg_dict['issues'])}")
                reason = ", ".join(reason_parts) if reason_parts else "removed during orchestration"
                print(f"  ❌ [Clip {seg_dict['clip_index'] + 1}] \"{seg_dict['text']}\"")
                print(f"      Reason: {reason}")
    else:
        print("  (No sentences removed)")

    print(f"\n📊 Summary:")
    print(f"  • Original: {result['stats']['original_count']} sentences")
    print(f"  • Final: {result['stats']['final_count']} sentences")
    print(f"  • Removed: {result['stats']['removed_count']} sentences")
    print(f"  • Final script: \"{result['final_script']}\"")


def compare_old_vs_new():
    """Compare old word-level vs new sentence-level approach."""
    print("\n\n" + "=" * 70)
    print("COMPARISON: Old vs New Approach")
    print("=" * 70)

    print("\n❌ OLD APPROACH (Word-Level):")
    print("  • Works at word/timestamp level")
    print("  • No sentence context")
    print("  • No quality assessment")
    print("  • Can't compare multiple takes")
    print("  • Model: Gemini 2.0 Flash (fast, basic)")
    print("\n  Example problem:")
    print('    Take 1: "I built the tool..." (correct grammar)')
    print('    Take 2: "I build the tool..." (incorrect grammar)')
    print("    ❌ Kept Take 2 (wrong choice)")

    print("\n✅ NEW APPROACH (Sentence-Level):")
    print("  • Works at sentence level")
    print("  • Full sentence context")
    print("  • Quality scoring (grammar, audio, completeness)")
    print("  • Compares all takes, selects best")
    print("  • Model: Gemini 2.0 Flash Thinking (smarter, reasoning)")
    print("\n  Same example:")
    print('    Take 1: "I built the tool..." (quality: 0.95)')
    print('    Take 2: "I build the tool..." (quality: 0.45 - grammar issue)')
    print("    ✅ Keeps Take 1 (correct choice)")


if __name__ == "__main__":
    print("\n🧪 SCRIPT ORCHESTRATION TEST SUITE\n")

    # This test suite requires the Gemini API key.
    if not os.getenv("GEMINI_API_KEY"):
        print("⚠️  GEMINI_API_KEY not set; skipping orchestration tests.")
        print("   Set it and re-run to execute the full suite.")
        sys.exit(0)

    # Check if transcripts exist
    if not Path("output/Ricky 1_edited_transcript.json").exists():
        print("❌ No transcripts found in output/")
        print("   Please run the pipeline first:")
        print("   python3 editor_agent.py 'assests/Ricky 1.mov' 'assests/Ricky 2.mov' 'assests/Ricky 3.mov' --agentic")
        sys.exit(1)

    try:
        # Run tests
        test_single_clip()
        test_multi_clip()
        compare_old_vs_new()

        print("\n\n" + "=" * 70)
        print("✅ ALL TESTS COMPLETE")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
