from ultralytics import YOLO
import torch

class ObjectDetector:
    def __init__(self, model_name="yolov8n.pt", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model = YOLO(model_name)
        self.model.to(self.device)

    def detect(self, frame):
        results = self.model(frame, verbose=False)[0]
        detections = []

        for box in results.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())

            if cls_id != 0:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w = x2 - x1
            h = y2 - y1

            detections.append([[x1, y1, w, h], conf, cls_id])

        return detections