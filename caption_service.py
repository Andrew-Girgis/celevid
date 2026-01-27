import subprocess
import pysubs2
import tempfile
from pathlib import Path
from typing import Dict, List


class CaptionService:
    """Service for adding captions to videos using FFmpeg"""

    STYLES = {
        "default": {
            "Alignment": pysubs2.Alignment.BOTTOM_CENTER,
            "FontName": "Arial",
            "FontSize": 24,
            "PrimaryColour": pysubs2.Color(255, 255, 255, 0),  # White
            "OutlineColour": pysubs2.Color(0, 0, 0, 0),  # Black
            "BorderStyle": 3,
            "Outline": 2,
            "Shadow": 1,
            "MarginV": 40,
            "Bold": True
        },
        "tiktok": {
            "Alignment": pysubs2.Alignment.BOTTOM_CENTER,
            "FontName": "Impact",  # Fallback from Montserrat if not available
            "FontSize": 32,
            "PrimaryColour": pysubs2.Color(255, 255, 255, 0),
            "OutlineColour": pysubs2.Color(0, 0, 0, 0),
            "BorderStyle": 4,
            "Outline": 3,
            "Shadow": 2,
            "MarginV": 60,
            "Bold": True
        },
        "mrbeast": {
            "Alignment": pysubs2.Alignment.BOTTOM_CENTER,
            "FontName": "Impact",
            "FontSize": 42,
            "PrimaryColour": pysubs2.Color(255, 255, 0, 0),  # Yellow
            "OutlineColour": pysubs2.Color(0, 0, 0, 0),
            "BorderStyle": 3,
            "Outline": 5,
            "Shadow": 4,
            "MarginV": 70,
            "Bold": True
        },
        "professional": {
            "Alignment": pysubs2.Alignment.BOTTOM_CENTER,
            "FontName": "Helvetica",
            "FontSize": 20,
            "PrimaryColour": pysubs2.Color(255, 255, 255, 0),
            "BackColour": pysubs2.Color(0, 0, 0, 128),  # Semi-transparent black
            "BorderStyle": 4,
            "MarginV": 30,
            "Bold": False
        }
    }

    def add_captions(
        self,
        video_path: str,
        transcript: Dict,
        output_path: str,
        style: str = "tiktok",
        words_per_caption: int = 1,
        caption_mode: str = "word_by_word"
    ) -> str:
        """
        Add captions to video

        Args:
            video_path: Path to input video
            transcript: Whisper transcript with chunks
            output_path: Path for output video
            style: Caption style ("default", "tiktok", "mrbeast", "professional")
            words_per_caption: Number of words to group per caption (used in segment mode)
            caption_mode: "word_by_word" (each word appears only during its timestamp)
                         or "segment" (words accumulate as they're said)

        Returns:
            Path to output video
        """
        # 1. Generate SRT file
        srt_path = self._generate_srt(transcript, words_per_caption, caption_mode)

        # 2. Get basic style config for drawtext
        style_dict = self.STYLES.get(style, self.STYLES["default"])

        # Convert pysubs2 Color to FFmpeg hex color
        primary_color = style_dict.get("PrimaryColour", pysubs2.Color(255, 255, 255, 0))
        outline_color = style_dict.get("OutlineColour", pysubs2.Color(0, 0, 0, 0))

        # FFmpeg color format is hex: 0xRRGGBB
        text_color = f"0x{primary_color.r:02X}{primary_color.g:02X}{primary_color.b:02X}"
        border_color = f"0x{outline_color.r:02X}{outline_color.g:02X}{outline_color.b:02X}"

        # Build simple subtitles filter (no fancy styling, but works reliably)
        try:
            # Save SRT to a fixed location to avoid path issues
            final_srt = Path(output_path).parent / "temp_subtitles.srt"
            Path(srt_path).rename(final_srt)

            subprocess.run([
                "ffmpeg",
                "-i", video_path,
                "-vf", f"subtitles={str(final_srt)}",
                "-c:a", "copy",
                "-y",
                output_path
            ], check=True, capture_output=True, text=True)

        except subprocess.CalledProcessError as e:
            raise Exception(f"FFmpeg error: {e.stderr}")
        finally:
            # Cleanup
            final_srt.unlink(missing_ok=True)

        return output_path

    def _generate_ass(self, transcript: Dict, words_per_caption: int, style: str, caption_mode: str = "word_by_word") -> str:
        """
        Convert Whisper transcript to ASS file with embedded styles

        Args:
            transcript: Dict with 'chunks' containing timestamp/text pairs
            words_per_caption: Words to group together (used in segment mode)
            style: Style name to apply
            caption_mode: "word_by_word" or "segment"

        Returns:
            Path to temporary ASS file
        """
        subs = pysubs2.SSAFile()

        # Apply style configuration
        style_dict = self.STYLES.get(style, self.STYLES["default"])

        # Create a style for the subtitles with proper attributes
        subs_style = pysubs2.SSAStyle()
        subs_style.fontname = style_dict.get("FontName", "Arial")
        subs_style.fontsize = style_dict.get("FontSize", 24)
        subs_style.primary_color = style_dict.get("PrimaryColour", pysubs2.Color(255, 255, 255, 0))
        subs_style.outline_color = style_dict.get("OutlineColour", pysubs2.Color(0, 0, 0, 0))
        subs_style.outline = style_dict.get("Outline", 2)
        subs_style.shadow = style_dict.get("Shadow", 1)
        subs_style.alignment = style_dict.get("Alignment", pysubs2.Alignment.BOTTOM_CENTER)
        subs_style.marginv = style_dict.get("MarginV", 40)
        subs_style.bold = style_dict.get("Bold", True)

        if "BackColour" in style_dict:
            subs_style.back_color = style_dict["BackColour"]

        subs.styles["Default"] = subs_style

        chunks = transcript.get("chunks", [])

        if caption_mode == "word_by_word":
            # Each word appears ONLY during its timestamp window
            for chunk in chunks:
                text = chunk.get("text", "").strip()
                if not text:
                    continue

                chunk_start, chunk_end = chunk["timestamp"]

                event = pysubs2.SSAEvent(
                    start=int(chunk_start * 1000),
                    end=int(chunk_end * 1000),
                    text=text
                )
                subs.append(event)

        else:  # segment mode
            # Words accumulate and stay visible until segment ends
            current_words = []
            start_time = None

            for chunk in chunks:
                text = chunk.get("text", "").strip()
                if not text:
                    continue

                chunk_start, chunk_end = chunk["timestamp"]

                if start_time is None:
                    start_time = chunk_start

                current_words.append(text)

                if len(current_words) >= words_per_caption:
                    event = pysubs2.SSAEvent(
                        start=int(start_time * 1000),
                        end=int(chunk_end * 1000),
                        text=" ".join(current_words)
                    )
                    subs.append(event)

                    current_words = []
                    start_time = None

            # Handle remaining words
            if current_words and chunks:
                last_chunk = chunks[-1]
                event = pysubs2.SSAEvent(
                    start=int(start_time * 1000),
                    end=int(last_chunk["timestamp"][1] * 1000),
                    text=" ".join(current_words)
                )
                subs.append(event)

        # Save to temporary ASS file
        ass_path = tempfile.mktemp(suffix=".ass")
        subs.save(ass_path)
        return ass_path

    def _generate_srt(self, transcript: Dict, words_per_caption: int, caption_mode: str = "word_by_word") -> str:
        """
        Convert Whisper transcript to SRT file

        Args:
            transcript: Dict with 'chunks' containing timestamp/text pairs
            words_per_caption: Words to group together (used in segment mode)
            caption_mode: "word_by_word" or "segment"

        Returns:
            Path to temporary SRT file
        """
        subs = pysubs2.SSAFile()
        chunks = transcript.get("chunks", [])

        if caption_mode == "word_by_word":
            # Each word appears ONLY during its timestamp window
            for chunk in chunks:
                text = chunk.get("text", "").strip()
                if not text:
                    continue

                chunk_start, chunk_end = chunk["timestamp"]

                event = pysubs2.SSAEvent(
                    start=int(chunk_start * 1000),
                    end=int(chunk_end * 1000),
                    text=text
                )
                subs.append(event)

        else:  # segment mode
            # Words accumulate and stay visible until segment ends
            current_words = []
            start_time = None

            for chunk in chunks:
                text = chunk.get("text", "").strip()
                if not text:
                    continue

                chunk_start, chunk_end = chunk["timestamp"]

                if start_time is None:
                    start_time = chunk_start

                current_words.append(text)

                # Create caption when we reach word limit
                if len(current_words) >= words_per_caption:
                    event = pysubs2.SSAEvent(
                        start=int(start_time * 1000),
                        end=int(chunk_end * 1000),
                        text=" ".join(current_words)
                    )
                    subs.append(event)

                    # Reset for next caption
                    current_words = []
                    start_time = None

            # Handle remaining words
            if current_words and chunks:
                last_chunk = chunks[-1]
                event = pysubs2.SSAEvent(
                    start=int(start_time * 1000),
                    end=int(last_chunk["timestamp"][1] * 1000),
                    text=" ".join(current_words)
                )
                subs.append(event)

        # Save to temporary file
        srt_path = tempfile.mktemp(suffix=".srt")
        subs.save(srt_path)
        return srt_path

    def _dict_to_style_string(self, style_dict: Dict[str, str]) -> str:
        """Convert style dictionary to FFmpeg style string"""
        return ",".join(f"{k}={v}" for k, v in style_dict.items())


# Standalone function for simple use
def add_subtitles_ffmpeg(video_path: str, srt_path: str, output_path: str, style: str = None):
    """
    Burn subtitles into video using FFmpeg (hardware accelerated)

    Args:
        video_path: Input video file path
        srt_path: SRT subtitle file path
        output_path: Output video file path
        style: Optional style string (FFmpeg format)
    """
    # Default TikTok/IG Reels style
    if style is None:
        style = (
            "Alignment=10,"
            "FontName=Impact,"
            "FontSize=24,"
            "PrimaryColour=&H00FFFFFF,"
            "OutlineColour=&H00000000,"
            "BorderStyle=3,"
            "Outline=2,"
            "Shadow=1,"
            "MarginV=40"
        )

    subprocess.run([
        "ffmpeg",
        "-i", video_path,
        "-vf", f"subtitles={srt_path}:force_style='{style}'",
        "-c:a", "copy",
        "-y",
        output_path
    ], check=True)
