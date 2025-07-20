import subprocess
import os

def save_scenes_ffmpeg(video_path, scenes, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for idx, scene in enumerate(scenes):
        start = scene["start"]
        duration = scene["end"] - scene["start"]
        output_path = f"{output_dir}/scene_{idx}.mp4"

        cmd = [
            "ffmpeg",
            "-y",  # overwrite if exists
            "-i", video_path,
            "-ss", str(start),
            "-t", str(duration),
            "-c:v", "h264_nvenc",  # hardware-accelerated encoder if available
            "-c:a", "aac",
            output_path
        ]

        subprocess.run(cmd)