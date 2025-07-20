from deep_sort_realtime.deepsort_tracker import DeepSort

class DeepSortTracker:
    def __init__(self):
        self.tracker = DeepSort(max_age=30)

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