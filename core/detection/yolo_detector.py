from ultralytics import YOLO
from .base_detector import BaseDetector
from config.settings import YOLO_CONFIG


class YOLODetector(BaseDetector):
    def __init__(self, model_path: str, conf: float | None = None):
        self.model = YOLO(model_path)
        self.conf = conf if conf is not None else YOLO_CONFIG["confidence_threshold"]

    def detect(self, image):
        results = self.model(image, conf=self.conf)
        detections = []

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()

            for box, score, cls in zip(boxes, scores, classes):
                detections.append({
                    "bbox": box.tolist(),
                    "confidence": float(score),
                    "class_id": int(cls),
                })

        return detections