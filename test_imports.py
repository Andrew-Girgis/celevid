#!/usr/bin/env python3
"""
Quick import test to verify all modules load correctly.
Run this to catch any import errors before running the full pipeline.
"""

import sys

def test_imports():
    """Test that all new modules can be imported."""
    print("Testing imports...")

    try:
        print("  ✓ Importing command_interpreter...")
        from command_interpreter import CommandInterpreter

        print("  ✓ Importing session_state...")
        from session_state import EditingSession

        print("  ✓ Importing agentic_pipeline...")
        from agentic_pipeline import AgenticVideoEditor

        print("  ✓ Importing editor_agent...")
        import editor_agent

        print("\n✅ All imports successful!")
        return True

    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False


def test_class_instantiation():
    """Test that classes can be instantiated (without API key)."""
    print("\nTesting class instantiation...")

    try:
        print("  ✓ Creating EditingSession...")
        from session_state import EditingSession
        session = EditingSession(["video1.mp4", "video2.mp4"], "output")

        print(f"    - Initial clip order: {session.clip_order}")
        print(f"    - Output dir: {session.output_dir}")

        print("\n✅ Basic instantiation successful!")
        return True

    except Exception as e:
        print(f"\n❌ Error during instantiation: {e}")
        return False


if __name__ == "__main__":
    success = test_imports() and test_class_instantiation()

    if success:
        print("\n" + "=" * 70)
        print("🎉 All tests passed! The agentic pipeline is ready to use.")
        print("=" * 70)
        print("\nNext steps:")
        print("1. Set GEMINI_API_KEY environment variable")
        print("2. Run: python3 editor_agent.py video1.mov video2.mov --agentic")
        sys.exit(0)
    else:
        print("\n" + "=" * 70)
        print("❌ Some tests failed. Please check the errors above.")
        print("=" * 70)
        sys.exit(1)
