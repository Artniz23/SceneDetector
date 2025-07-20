from scene_splitter.clip_scene_splitter import CLIPSceneSplitter
from tracking.detector import ObjectDetector
from tracking.tracker import DeepSortTracker
from utils.video_tools import save_scenes_ffmpeg
import cv2


def analyze_video(video_path, scene_splitter, detector, tracker, every_n_frames=15):
    scenes = scene_splitter.detect_scenes(video_path, every_n_frames=every_n_frames)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    scene_data = []

    for (start, end) in scenes:
        cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
        frame_idx = int(start * fps)
        end_idx = int(end * fps)

        track_ids_in_scene = set()

        while frame_idx < end_idx:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % every_n_frames == 0:
                detections = detector.detect(frame)
                tracks = tracker.update(detections, frame)

                for t in tracks:
                    track_ids_in_scene.add(t['track_id'])

            frame_idx += 1

        scene_data.append({
            "start": start,
            "end": end,
            "track_ids": list(track_ids_in_scene)
        })

    cap.release()
    return scene_data


def group_by_characters(scenes, threshold=0.5):
    grouped = []
    current_group = [scenes[0]]

    def jaccard(a, b):
        a_set, b_set = set(a), set(b)
        intersection = a_set & b_set
        union = a_set | b_set
        return len(intersection) / len(union) if union else 0

    for i in range(1, len(scenes)):
        prev = current_group[-1]
        curr = scenes[i]

        sim = jaccard(prev["track_ids"], curr["track_ids"])
        if sim >= threshold:
            current_group.append(curr)
        else:
            grouped.append({
                "start": current_group[0]["start"],
                "end": current_group[-1]["end"],
                "track_ids": list(set().union(*(s["track_ids"] for s in current_group)))
            })
            current_group = [curr]

    # don't forget last group
    if current_group:
        grouped.append({
            "start": current_group[0]["start"],
            "end": current_group[-1]["end"],
            "track_ids": list(set().union(*(s["track_ids"] for s in current_group)))
        })

    return grouped

def main():
    video_path = "test.mp4"
    splitter = CLIPSceneSplitter()
    scenes = splitter.detect_scenes(video_path)
    # detector = ObjectDetector()
    # tracker = DeepSortTracker()
    # video_path = "test.mp4"
    #
    # scenes = analyze_video(video_path, splitter, detector, tracker)
    #
    # scenes = group_by_characters(scenes)
    #
    # output_dir = "output_scenes"
    #
    # print(scenes)
    #
    # save_scenes_ffmpeg(video_path, scenes, output_dir)

    return 0

if __name__ == "__main__":
    main()
