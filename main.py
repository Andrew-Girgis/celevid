from transformers import pipeline
import torch
import librosa
import sys

# Auto-detect best device: CUDA (NVIDIA GPU) > MPS (Apple Metal) > CPU
if torch.cuda.is_available():
    device = "cuda"
    print("🚀 Using NVIDIA GPU (CUDA)")
elif torch.backends.mps.is_available():
    device = "mps"
    print("🚀 Using Apple Metal GPU (MPS)")
else:
    device = "cpu"
    print("⚠️  Using CPU (slower)")

pipe = pipeline(
    "automatic-speech-recognition",
    model="distil-whisper/distil-large-v3",
    dtype=torch.float16,
    device=device,
)

# Get audio file from command line or use default
audio_file_path = sys.argv[1] if len(sys.argv) > 1 else "assets/Ricky 1.mov"

print(f"Loading audio from {audio_file_path}...")
# Use librosa to load audio (supports m4a, mov, and many other formats via ffmpeg)
audio_array, sampling_rate = librosa.load(audio_file_path, sr=16000)

# Trim silence from the beginning and end
audio_trimmed, trim_indices = librosa.effects.trim(audio_array, top_db=30)
# Calculate the offset in seconds where speech actually starts
start_offset = trim_indices[0] / sampling_rate

print(f"Transcribing using device: {device}")
print(f"Speech detected starting at {start_offset:.2f} seconds")
result = pipe(audio_trimmed, return_timestamps="word")

print("\nTranscription:")
print(result["text"])

if result.get("chunks"):
    print("\nWord-level timestamps (adjusted to original audio):")
    for chunk in result["chunks"]:
        # Adjust timestamps by adding the start offset
        start_time = chunk['timestamp'][0] + start_offset if chunk['timestamp'][0] is not None else None
        end_time = chunk['timestamp'][1] + start_offset if chunk['timestamp'][1] is not None else None
        print(f"({start_time:.2f}, {end_time:.2f}): {chunk['text']}")