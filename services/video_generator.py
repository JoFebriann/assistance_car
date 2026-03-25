import cv2
import numpy as np
from utils.frame_models import FrameData
from utils.logger import get_logger
from config.settings import PROCESSING_CONFIG


class VideoFrameGenerator:

    def __init__(self, video_path):
        self.logger = get_logger("MP4Generator")
        self.video_path = video_path
        # Updated to actual video FPS inside generate() before the first yield
        self.fps = PROCESSING_CONFIG["default_fps"]

    def generate(self):
        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")

        self.fps = cap.get(cv2.CAP_PROP_FPS) or PROCESSING_CONFIG["default_fps"]

        frame_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR → RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # MP4 tidak punya depth
            depth = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.float32)

            # Dummy camera matrix
            camera_matrix = np.eye(3)

            yield FrameData(
                frame_id=frame_id,
                timestamp=cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0,
                rgb_image=rgb,
                depth_map=depth,
                camera_matrix=camera_matrix,
                image_path=None,
                depth_path=None
            )

            frame_id += 1

        cap.release()
        self.logger.info("MP4 processing finished.")