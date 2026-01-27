"""
Audio extraction and processing utilities
"""
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple


class AudioProcessor:
    """Handles audio extraction and format conversion."""

    @staticmethod
    def extract_audio_to_m4a(video_path: str, output_path: str) -> str:
        """
        Extract audio from video to M4A format (high quality).

        Args:
            video_path: Input video file
            output_path: Output M4A file path

        Returns:
            Path to extracted M4A file
        """
        subprocess.run([
            "ffmpeg",
            "-i", video_path,
            "-vn",  # No video
            "-acodec", "aac",  # AAC codec
            "-b:a", "256k",  # High quality bitrate
            "-y",
            output_path
        ], check=True, capture_output=True, text=True)

        return output_path

    @staticmethod
    def convert_to_wav(audio_path: str, temp_dir: str = None) -> Tuple[str, bool]:
        """
        Convert audio to WAV for transcription processing.

        Args:
            audio_path: Input audio file (any format)
            temp_dir: Optional temp directory

        Returns:
            Tuple of (wav_path, is_temp_file)
        """
        audio_path = Path(audio_path)

        # If already WAV, use as-is
        if audio_path.suffix.lower() == '.wav':
            return str(audio_path), False

        # Convert to WAV in temp location
        if temp_dir:
            Path(temp_dir).mkdir(exist_ok=True)
            wav_path = Path(temp_dir) / f"{audio_path.stem}.wav"
        else:
            wav_path = tempfile.mktemp(suffix=".wav")

        subprocess.run([
            "ffmpeg",
            "-i", str(audio_path),
            "-ar", "16000",  # 16kHz for Whisper
            "-ac", "1",  # Mono
            "-c:a", "pcm_s16le",  # 16-bit PCM
            "-y",
            str(wav_path)
        ], check=True, capture_output=True, text=True)

        return str(wav_path), True

    @staticmethod
    def get_audio_info(file_path: str) -> dict:
        """Get audio/video file information."""
        result = subprocess.run([
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            file_path
        ], capture_output=True, text=True, check=True)

        import json
        return json.loads(result.stdout)

    @staticmethod
    def has_audio(video_path: str) -> bool:
        """Check if file has audio stream."""
        info = AudioProcessor.get_audio_info(video_path)
        streams = info.get("streams", [])
        return any(s.get("codec_type") == "audio" for s in streams)


class VideoQualityPresets:
    """Quality presets for video encoding."""

    PRESETS = {
        "fast": {
            "preset": "ultrafast",
            "crf": "28",  # Lower quality, faster
            "profile": "baseline"
        },
        "balanced": {
            "preset": "medium",
            "crf": "23",  # Balanced
            "profile": "main"
        },
        "quality": {
            "preset": "slow",
            "crf": "18",  # High quality
            "profile": "high"
        },
        "maximum": {
            "preset": "veryslow",
            "crf": "15",  # Maximum quality
            "profile": "high"
        }
    }

    @staticmethod
    def get_encoding_params(quality: str = "balanced") -> list:
        """
        Get FFmpeg encoding parameters for quality preset.

        Args:
            quality: Preset name (fast, balanced, quality, maximum)

        Returns:
            List of FFmpeg parameters
        """
        preset = VideoQualityPresets.PRESETS.get(quality, VideoQualityPresets.PRESETS["balanced"])

        return [
            "-c:v", "libx264",  # H.264 codec
            "-preset", preset["preset"],
            "-crf", preset["crf"],
            "-profile:v", preset["profile"],
            "-pix_fmt", "yuv420p",  # Compatible pixel format
            "-movflags", "+faststart",  # Web streaming optimization
        ]


def normalize_video_to_mp4(
    input_path: str,
    output_path: str,
    quality: str = "balanced"
) -> str:
    """
    Normalize any video format to standardized MP4.

    This ensures consistent format for all processing steps:
    - Converts any format (MOV, AVI, MKV, etc.) to MP4
    - Applies quality preset encoding
    - Standardizes audio to AAC
    - Optimizes for web streaming

    Args:
        input_path: Input video (any format)
        output_path: Output MP4 path
        quality: Quality preset to use

    Returns:
        Path to normalized MP4
    """
    cmd = ["ffmpeg", "-i", input_path]

    # Video encoding with quality preset
    cmd.extend(VideoQualityPresets.get_encoding_params(quality))

    # Audio: AAC at good quality
    cmd.extend(["-c:a", "aac", "-b:a", "192k"])

    cmd.extend(["-y", output_path])

    subprocess.run(cmd, check=True, capture_output=True, text=True)

    return output_path


def process_video_with_quality(
    input_path: str,
    output_path: str,
    quality: str = "balanced",
    video_filter: str = None,
    copy_audio: bool = True
):
    """
    Re-encode video with quality settings.

    Args:
        input_path: Input video
        output_path: Output video
        quality: Quality preset
        video_filter: Optional video filter string (e.g., subtitles)
        copy_audio: If True, copy audio without re-encoding
    """
    cmd = ["ffmpeg", "-i", input_path]

    # Video encoding
    if video_filter:
        cmd.extend(["-vf", video_filter])

    cmd.extend(VideoQualityPresets.get_encoding_params(quality))

    # Audio handling
    if copy_audio:
        cmd.extend(["-c:a", "copy"])
    else:
        cmd.extend(["-c:a", "aac", "-b:a", "192k"])

    cmd.extend(["-y", output_path])

    subprocess.run(cmd, check=True, capture_output=True, text=True)
