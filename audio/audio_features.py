import whisper
import librosa
import numpy as np

def transcribe_audio_whisper(audio_path, model_size="tiny"):
    model = whisper.load_model(model_size, device="cuda")
    result = model.transcribe(audio_path)
    # TODO
    return result
    # return result['segments']


def get_audio_activity(audio_path, frame_duration=1.0):
    y, sr = librosa.load(audio_path, sr=None)
    frame_length = int(sr * frame_duration)
    energy = [
        np.sqrt(np.mean(y[i:i+frame_length]**2))
        for i in range(0, len(y), frame_length)
    ]
    return energy

def enrich_scenes_with_audio(scenes, segments, energy, frame_duration=1.0):
    for scene in scenes:
        start, end = scene["start"], scene["end"]

        # Текст: учитываем пересекающиеся сегменты
        scene_texts = [
            seg["text"]
            for seg in segments
            if not (seg["end"] < start or seg["start"] > end)
        ]
        scene["transcript"] = " ".join(scene_texts).strip()

        # Энергия: защищённый доступ
        start_idx = max(0, int(start // frame_duration))
        end_idx = min(len(energy) - 1, int(end // frame_duration))
        scene_energy = energy[start_idx:end_idx+1] if end_idx >= start_idx else []
        scene["avg_rms"] = float(np.mean(scene_energy)) if scene_energy else 0.0

    return scenes