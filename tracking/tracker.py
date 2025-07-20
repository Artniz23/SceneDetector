from deep_sort_realtime.deepsort_tracker import DeepSort

class DeepSortTracker:
    def __init__(self):
        self.tracker = DeepSort(
            max_age=30,
            n_init=3,
            nms_max_overlap=1.0,
            max_cosine_distance=0.2,
            nn_budget=None,
            override_track_class=None,
            embedder="mobilenet",  # или osnet_x0_25 (точнее, но медленнее)
            half=True,  # использовать fp16 (для скорости)
            bgr=True,
            embedder_gpu=True  # <== Включает GPU для эмбеддера
        )

    def update(self, detections, frame):
        """
        detections: List of [x, y, w, h, conf]
        Returns: list of dicts: {track_id, bbox, ...}
        """
        tracks = self.tracker.update_tracks(detections, frame=frame)
        output = []

        for track in tracks:
            if not track.is_confirmed():
                continue
            bbox = track.to_ltrb()
            output.append({
                'track_id': track.track_id,
                'bbox': bbox,
            })

        return output