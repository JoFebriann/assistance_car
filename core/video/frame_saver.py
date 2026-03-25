import cv2
from pathlib import Path
from utils.logger import get_logger
from config.settings import TEMP_FRAMES_DIR, PROCESSING_CONFIG


class FrameSaver:

    def __init__(self, run_id):
        self.logger = get_logger("FrameSaver")
        self.run_id = run_id

        self.temp_dir = TEMP_FRAMES_DIR / run_id

        if PROCESSING_CONFIG["clean_temp_frames_on_start"] and self.temp_dir.exists():
            # overwrite setiap run
            import shutil
            shutil.rmtree(self.temp_dir)

        self.temp_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Frame temp directory: {self.temp_dir}")

    def save(self, frame_id, annotated_bgr):

        filename = self.temp_dir / f"frame_{frame_id:06d}.png"

        cv2.imwrite(str(filename), annotated_bgr)

    def get_frame_dir(self):
        return self.temp_dir