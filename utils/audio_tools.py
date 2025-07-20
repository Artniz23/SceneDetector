import subprocess

def extract_audio_ffmpeg(video_path, audio_path="audio.wav"):
    command = [
        "ffmpeg",
        "-y",  # overwrite
        "-i", video_path,
        "-vn",  # no video
        "-acodec", "pcm_s16le",  # raw WAV format
        "-ar", "16000",  # sample rate for whisper
        "-ac", "1",  # mono
        audio_path
    ]
    subprocess.run(command, check=True)
    return audio_path