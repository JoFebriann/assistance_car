from __future__ import annotations

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from config.settings import YOLO_CONFIG

from .base_detector import BaseDetector


class YOLODetector(BaseDetector):
    """Wrapper around ultralytics YOLO with optional device selection.

    device: None|'auto'|'cpu'|'cuda' or 'cuda:0' etc. If 'auto' or None,
    will pick CUDA when available.
    """

    def __init__(self, model_path: str, conf: float | None = None, device: str | None = None):
        # Resolve device
        resolved = device
        if resolved is None or resolved == "auto":
            resolved = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = resolved
        self.use_fp16 = bool(YOLO_CONFIG.get("fp16_enabled", False)) and str(self.device).startswith("cuda")

        # Load model and move to device
        self.model = YOLO(model_path)
        try:
            # attempt to move model to device if supported
            self.model.to(self.device)
        except Exception:
            # some ultralytics versions ignore .to(); rely on call-time device
            pass

        self.conf = conf if conf is not None else YOLO_CONFIG["confidence_threshold"]
        self.class_names = getattr(self.model, "names", {})
        self.input_resize_enabled = bool(YOLO_CONFIG.get("input_resize_enabled", False))
        self.input_width = int(YOLO_CONFIG.get("input_width", 416))
        self.input_height = int(YOLO_CONFIG.get("input_height", 320))

    def detect(self, image):
        orig_h, orig_w = image.shape[:2]
        scale_x = 1.0
        scale_y = 1.0
        inference_image = image

        if self.input_resize_enabled:
            inference_image = cv2.resize(
                image,
                (self.input_width, self.input_height),
                interpolation=cv2.INTER_LINEAR,
            )
            scale_x = orig_w / float(self.input_width)
            scale_y = orig_h / float(self.input_height)

        # Call model using the model's device if possible; pass device as fallback
        try:
            results = self.model(
                inference_image,
                conf=self.conf,
                device=self.device,
                half=self.use_fp16,
                verbose=False,
            )
        except TypeError:
            # older ultralytics versions may not support all predict kwargs
            try:
                results = self.model(inference_image, conf=self.conf, device=self.device, half=self.use_fp16)
            except TypeError:
                results = self.model(inference_image, conf=self.conf, device=self.device)

        detections = []

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()

            for box, score, cls in zip(boxes, scores, classes):
                if self.input_resize_enabled:
                    box = box.copy()
                    box[0] *= scale_x
                    box[2] *= scale_x
                    box[1] *= scale_y
                    box[3] *= scale_y
                    box[0] = float(np.clip(box[0], 0, orig_w - 1))
                    box[2] = float(np.clip(box[2], 0, orig_w - 1))
                    box[1] = float(np.clip(box[1], 0, orig_h - 1))
                    box[3] = float(np.clip(box[3], 0, orig_h - 1))

                class_id = int(cls)
                class_name = self.class_names.get(class_id, str(class_id))
                detections.append({
                    "bbox": box.tolist(),
                    "confidence": float(score),
                    "class_id": class_id,
                    "class_name": str(class_name),
                })

        return detections