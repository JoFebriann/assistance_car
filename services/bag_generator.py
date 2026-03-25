import pyrealsense2 as rs
import numpy as np
from utils.frame_models import FrameData
from utils.logger import get_logger
from config.settings import PROCESSING_CONFIG

MAX_FRAMES = PROCESSING_CONFIG["max_bag_frames"]

class BagFrameGenerator:

    def __init__(self, bag_path):
        self.logger = get_logger("BagGenerator")
        self.bag_path = bag_path
        # Updated to actual stream FPS inside generate() before the first yield
        self.fps = PROCESSING_CONFIG["default_fps"]

    def generate(self):

        self.logger.info(f"Opening BAG file: {self.bag_path}")

        pipeline = rs.pipeline()
        config = rs.config()

        config.enable_device_from_file(
            self.bag_path,
            repeat_playback=False
        )

        config.enable_stream(rs.stream.color)
        config.enable_stream(rs.stream.depth)

        profile = pipeline.start(config)

        # Disable real-time so ALL frames are read regardless of processing speed
        playback = profile.get_device().as_playback()
        playback.set_real_time(False)

        # Read actual recording FPS so video output plays at the correct speed
        try:
            self.fps = float(
                profile.get_stream(rs.stream.color)
                .as_video_stream_profile()
                .fps()
            )
            self.logger.info(f"BAG stream FPS: {self.fps}")
        except Exception:
            self.logger.warning(
                f"Could not read FPS from profile; defaulting to {PROCESSING_CONFIG['default_fps']}"
            )
            self.fps = PROCESSING_CONFIG["default_fps"]

        # Align depth frame into the color frame coordinate system so that
        # YOLO bounding boxes (computed on the color image) map correctly to
        # depth pixels when we compute per-object distances.
        align = rs.align(rs.stream.color)

        frame_id = 0

        try:
            while True:
                frames = pipeline.wait_for_frames()

                # Project depth onto color pixel grid
                aligned_frames = align.process(frames)

                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()

                if not color_frame or not depth_frame:
                    continue

                rgb = np.asanyarray(color_frame.get_data())

                # Convert raw uint16 depth → meters using the sensor's scale factor
                # (RealSense default: 0.001 m/unit, i.e. raw 5000 → 5.0 m)
                depth_raw = np.asanyarray(depth_frame.get_data())
                depth_scale = depth_frame.get_units()
                depth_meters = depth_raw.astype(np.float32) * depth_scale

                intr = color_frame.profile.as_video_stream_profile().intrinsics

                camera_matrix = np.array([
                    [intr.fx, 0, intr.ppx],
                    [0, intr.fy, intr.ppy],
                    [0, 0, 1]
                ])

                self.logger.info(f"BAG frame {frame_id} extracted")

                yield FrameData(
                    frame_id=frame_id,
                    timestamp=frames.get_timestamp(),
                    rgb_image=rgb,
                    depth_map=depth_meters,
                    camera_matrix=camera_matrix
                )

                frame_id += 1

                if frame_id >= MAX_FRAMES:
                    self.logger.warning(
                        f"Max frame limit reached ({MAX_FRAMES}). Stopping generator."
                    )
                    break

        except RuntimeError:
            self.logger.info("Reached end of BAG file.")

        finally:
            pipeline.stop()
            self.logger.info("BAG pipeline stopped.")