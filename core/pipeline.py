from core.detection.yolo_detector import YOLODetector
from core.depth.stereo_depth import StereoDepth
from core.optical_flow.global_flow import GlobalOpticalFlow
from core.lane.lane_detector import LaneDetector
from core.calculation.risk_engine import RiskEngine
from config.settings import LANE_CONFIG

from database.repository import (
    FrameRepository,
    DetectionRepository,
    OpticalFlowRepository,
    SceneRepository,
    LaneRepository
)

from utils.logger import get_logger

import cv2
import numpy as np


class PerceptionPipeline:

    def __init__(self, model_path: str):
        self.logger = get_logger("Pipeline")

        # Models
        self.detector = YOLODetector(model_path)
        self.depth = StereoDepth()
        self.flow = GlobalOpticalFlow()
        self.lane_detector = None
        if LANE_CONFIG["enabled"]:
            try:
                self.lane_detector = LaneDetector()
            except Exception as exc:
                self.logger.warning("Lane detector disabled due to init error: %s", exc)
        self.risk_engine = RiskEngine()

        # Repositories
        self.frame_repo = FrameRepository()
        self.det_repo = DetectionRepository()
        self.flow_repo = OpticalFlowRepository()
        self.scene_repo = SceneRepository()
        self.lane_repo = LaneRepository()

    def reset(self):
        self.flow.reset()
        self.logger.info("Pipeline state reset.")

    def process_frame(self, frame_data, frame_saver=None):

        frame_id = frame_data.frame_id
        self.logger.info(f"Processing frame {frame_id}")

        # 1️⃣ Save frame info
        self.frame_repo.insert(frame_data)

        # 2️⃣ Detection
        detections = self.detector.detect(frame_data.rgb_image)
        self.logger.info(
            f"[Frame {frame_id}] Detected {len(detections)} objects"
        )

        calculations = []

        for det in detections:

            distance = self.depth.compute_distance(
                frame_data.depth_map,
                det["bbox"]
            )

            risk = self.risk_engine.estimate_risk(distance)

            result = {
                "frame_id": frame_id,
                "class_id": det["class_id"],
                "confidence": det["confidence"],
                "bbox": det["bbox"],
                "distance_m": distance,
                "risk": risk
            }

            calculations.append(result)
            self.det_repo.insert(result)

        # 3️⃣ Optical Flow
        flow_stats = self.flow.compute(frame_data.rgb_image)
        self.flow_repo.insert(frame_id, flow_stats)

        if flow_stats is not None:
            self.logger.info(
                f"[Frame {frame_id}] Flow mean={flow_stats['mean_magnitude']:.4f}"
            )
        else:
            self.logger.info(f"[Frame {frame_id}] First frame (no flow)")

        # 4️⃣ Lane Segmentation
        lane_result = None
        if self.lane_detector is not None:
            lane_result = self.lane_detector.detect(frame_data.rgb_image)
            self.lane_repo.insert(frame_id, lane_result)
            self.logger.info(
                f"[Frame {frame_id}] Lane ratio={lane_result['lane_pixel_ratio']:.4f}, "
                f"Drivable ratio={lane_result['drivable_pixel_ratio']:.4f}"
            )

        # 5️⃣ Scene Risk
        scene_risk = self.risk_engine.compute_scene_risk(calculations)
        alert_flag = 1 if scene_risk > 0 else 0

        self.scene_repo.insert(frame_id, scene_risk, alert_flag)

        self.logger.info(
            f"[Frame {frame_id}] Scene risk={scene_risk}"
        )

        # 6️⃣ Save Annotated Frame 
        if frame_saver is not None:

            self.logger.info(
                f"[Frame {frame_id}] Saving annotated frame"
            )

            annotated_frame = self._draw_annotations(
                frame_data.rgb_image,
                calculations,
                flow_stats,
                scene_risk,
                lane_result,
            )

            frame_saver.save(frame_id, annotated_frame)

        return {
            "frame_id": frame_id,
            "detections": calculations,
            "flow": flow_stats,
            "lane": lane_result,
            "scene_risk": scene_risk,
            "alert": alert_flag == 1
        }

    def _draw_annotations(self, rgb_image, calculations, flow_stats, scene_risk, lane_result=None):

        frame_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        if lane_result is not None:
            da_mask = lane_result.get("da_mask")
            ll_mask = lane_result.get("ll_mask")

            if da_mask is not None:
                da_color = np.array([0, 180, 0], dtype=np.float32)
                frame_bgr[da_mask == 1] = (
                    frame_bgr[da_mask == 1] * 0.6 + da_color * 0.4
                ).astype("uint8")

            if ll_mask is not None:
                ll_color = np.array([0, 220, 255], dtype=np.float32)
                frame_bgr[ll_mask == 1] = (
                    frame_bgr[ll_mask == 1] * 0.35 + ll_color * 0.65
                ).astype("uint8")

        for det in calculations:
            x1, y1, x2, y2 = map(int, det["bbox"])

            color = (0, 255, 0)
            if det["risk"] == "HIGH":
                color = (0, 0, 255)
            elif det["risk"] == "MEDIUM":
                color = (0, 255, 255)

            distance = det.get("distance_m", None)

            if distance is None:
                distance_text = "N/A"
            else:
                distance_text = f"{distance:.2f}m"

            label = f"{det['class_id']} | {distance_text} | {det['risk']}"

            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame_bgr,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )

        if flow_stats:
            cv2.putText(
                frame_bgr,
                f"Flow: {flow_stats['mean_magnitude']:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2
            )

        cv2.putText(
            frame_bgr,
            f"Scene Risk: {scene_risk}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

        if lane_result is not None:
            cv2.putText(
                frame_bgr,
                f"Lane: {lane_result['lane_pixel_ratio'] * 100:.1f}% | Drivable: {lane_result['drivable_pixel_ratio'] * 100:.1f}%",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (64, 64, 255),
                2,
            )

        return frame_bgr