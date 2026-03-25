from __future__ import annotations

import time
from typing import Any, Iterator

import numpy as np

from config.settings import PROCESSING_CONFIG, REALSENSE_CONFIG
from core.pipeline import PerceptionPipeline
from database.db import reset_database
from utils.frame_models import FrameData
from utils.logger import get_logger


class RealSenseRealtimeService:
    """Capture RealSense frames, process them, and stream annotated MJPEG."""

    def __init__(self, model_path: str):
        self.logger = get_logger("RealSenseRealtimeService")
        self.pipeline = PerceptionPipeline(model_path)

    def stream(self) -> Iterator[bytes]:
        try:
            import cv2  # type: ignore
            import pyrealsense2 as rs_module  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "pyrealsense2 and opencv-python are required for realtime mode."
            ) from exc

        rs: Any = rs_module

        rs_pipeline = rs.pipeline()
        rs_config = rs.config()

        width = REALSENSE_CONFIG["width"]
        height = REALSENSE_CONFIG["height"]
        camera_fps = REALSENSE_CONFIG["fps"]

        rs_config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, camera_fps)
        rs_config.enable_stream(rs.stream.depth, width, height, rs.format.z16, camera_fps)

        align = rs.align(rs.stream.color)

        reset_database()
        self.pipeline.reset()

        frame_id = 0
        target_fps = float(REALSENSE_CONFIG["target_process_fps"])
        frame_timeout_ms = int(REALSENSE_CONFIG["frame_timeout_ms"])
        jpeg_quality = int(REALSENSE_CONFIG["jpeg_quality"])

        self.logger.info("Starting RealSense realtime processing stream")
        started = False

        try:
            rs_pipeline.start(rs_config)
            started = True
            while True:
                tick = time.perf_counter()

                frames = rs_pipeline.wait_for_frames(timeout_ms=frame_timeout_ms)
                aligned = align.process(frames)

                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue

                bgr = np.asanyarray(color_frame.get_data())
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

                depth_raw = np.asanyarray(depth_frame.get_data())
                depth_scale = depth_frame.get_units()
                depth_meters = depth_raw.astype(np.float32) * depth_scale

                intr = color_frame.profile.as_video_stream_profile().intrinsics
                camera_matrix = np.array(
                    [
                        [intr.fx, 0, intr.ppx],
                        [0, intr.fy, intr.ppy],
                        [0, 0, 1],
                    ],
                    dtype=np.float32,
                )

                frame_data = FrameData(
                    frame_id=frame_id,
                    timestamp=time.time(),
                    rgb_image=rgb,
                    depth_map=depth_meters,
                    camera_matrix=camera_matrix,
                )

                result = self.pipeline.process_frame(frame_data)

                annotated = self.pipeline._draw_annotations(
                    rgb_image=frame_data.rgb_image,
                    calculations=result["detections"],
                    flow_stats=result["flow"],
                    scene_risk=result["scene_risk"],
                    lane_result=result.get("lane"),
                )

                ok, encoded = cv2.imencode(
                    ".jpg",
                    annotated,
                    [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
                )
                if not ok:
                    continue

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + encoded.tobytes()
                    + b"\r\n"
                )

                frame_id += 1

                if target_fps > 0:
                    elapsed = time.perf_counter() - tick
                    delay = (1.0 / target_fps) - elapsed
                    if delay > 0:
                        time.sleep(delay)

                max_frames = int(PROCESSING_CONFIG["max_bag_frames"])
                if frame_id >= max_frames:
                    self.logger.warning(
                        "Realtime stream reached max frame limit (%s), stopping.",
                        max_frames,
                    )
                    break

        except GeneratorExit:
            self.logger.info("Realtime stream client disconnected")
        finally:
            if started:
                rs_pipeline.stop()
            self.logger.info("RealSense realtime stream stopped")