from core.detection.yolo_detector import YOLODetector
from core.depth.stereo_depth import StereoDepth
from core.optical_flow.global_flow import GlobalOpticalFlow
from core.optical_flow.object_flow import ObjectOpticalFlow
from core.lane.lane_detector import LaneDetector
from core.calculation.risk_engine import RiskEngine
from config.settings import LANE_CONFIG

from database.repository import (
    FrameRepository,
    DetectionRepository,
    OpticalFlowRepository,
    SceneRepository,
    LaneRepository,
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
        self.object_flow = ObjectOpticalFlow()   # stateless — slices flow ROI per bbox
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
        self.logger.info(f"[Frame {frame_id}] Detected {len(detections)} objects")

        # 3️⃣ Optical Flow — global scene level
        #    compute() returns (stats_dict, flow_field) or None on the first frame.
        #    flow_field is a fresh (H, W, 2) numpy array for *this* frame pair;
        #    it is never stored on GlobalOpticalFlow — we pass it forward explicitly.
        flow_result = self.flow.compute(frame_data.rgb_image)
        flow_stats: dict | None = None
        flow_field: np.ndarray | None = None

        if flow_result is not None:
            flow_stats, flow_field = flow_result  # unpack tuple

        self.flow_repo.insert(frame_id, flow_stats)

        if flow_stats is not None:
            self.logger.info(
                f"[Frame {frame_id}] Flow mean={flow_stats['mean_magnitude']:.4f}"
            )
        else:
            self.logger.info(f"[Frame {frame_id}] First frame (no flow)")

        # 4️⃣ Per-object optical flow
        #    ObjectOpticalFlow is stateless: it receives the fresh flow_field
        #    as an argument and slices it per bounding box.  No cached state.
        if flow_field is not None:
            object_flow_list = self.object_flow.compute_object_flows(
                flow_field, detections
            )
        else:
            # First frame — no previous frame to compare against
            object_flow_list = [None] * len(detections)

        # 5️⃣ Per-object: depth → object flow → risk
        calculations = []
        for det, obj_flow in zip(detections, object_flow_list):

            distance = self.depth.compute_distance(frame_data.depth_map, det["bbox"])
            risk = self.risk_engine.estimate_risk(distance, object_flow=obj_flow)

            if obj_flow:
                self.logger.info(
                    f"[Frame {frame_id}] {det['class_id']} mag={obj_flow['object_magnitude']:.2f} "
                    f"moving={obj_flow['is_moving']} risk={risk}"
                )

            result = {
                "frame_id": frame_id,
                "class_id": det["class_id"],
                "confidence": det["confidence"],
                "bbox": det["bbox"],
                "distance_m": distance,
                "risk": risk,
                "object_flow": obj_flow,  # None on first frame
            }

            calculations.append(result)
            self.det_repo.insert(result)

        # 6️⃣ Lane Segmentation
        lane_result = None
        if self.lane_detector is not None:
            lane_result = self.lane_detector.detect(frame_data.rgb_image)
            self.lane_repo.insert(frame_id, lane_result)
            self.logger.info(
                f"[Frame {frame_id}] Lane ratio={lane_result['lane_pixel_ratio']:.4f}, "
                f"Drivable ratio={lane_result['drivable_pixel_ratio']:.4f}"
            )

        # 7️⃣ Scene Risk
        scene_risk = self.risk_engine.compute_scene_risk(calculations)
        alert_flag = 1 if scene_risk > 0 else 0

        self.scene_repo.insert(frame_id, scene_risk, alert_flag)
        self.logger.info(f"[Frame {frame_id}] Scene risk={scene_risk}")

        # 8️⃣ Save Annotated Frame
        if frame_saver is not None:
            self.logger.info(f"[Frame {frame_id}] Saving annotated frame")
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
            "alert": alert_flag == 1,
        }

    # ── Annotation helpers ────────────────────────────────────────────────────

    @staticmethod
    def _risk_color(risk: str, is_moving: bool) -> tuple:
        """
        Return BGR color for a bounding box.

        Stationary objects use solid primary colours so they stand out.
        Moving objects use a slightly muted blue-shifted tone to signal
        their reduced danger level at a glance.
        """
        if risk == "HIGH":
            return (0, 60, 220) if is_moving else (0, 0, 255)    # blue-red vs pure red
        if risk == "MEDIUM":
            return (30, 200, 255) if is_moving else (0, 215, 255) # sky vs yellow
        return (80, 200, 80)   # green — LOW is always low risk

    def _draw_annotations(
        self,
        rgb_image: np.ndarray,
        calculations: list,
        flow_stats: dict | None,
        scene_risk: int,
        lane_result: dict | None = None,
    ) -> np.ndarray:

        frame_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        # ── Lane / drivable overlay ──────────────────────────────────────────
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

        # ── Per-object annotations ───────────────────────────────────────────
        for det in calculations:
            x1, y1, x2, y2 = map(int, det["bbox"])
            obj_flow = det.get("object_flow")
            is_moving = bool(obj_flow and obj_flow.get("is_moving"))

            color = self._risk_color(det["risk"], is_moving)

            # Bounding box
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)

            # Distance text
            distance = det.get("distance_m")
            dist_text = f"{distance:.2f}m" if distance is not None else "N/A"

            # MOV / STA + magnitude badge
            if obj_flow is not None:
                motion_tag = f"MOV {obj_flow['object_magnitude']:.1f}px" if is_moving else "STA"
            else:
                motion_tag = ""

            label = f"{det['class_id']} | {dist_text} | {det['risk']}"
            if motion_tag:
                label += f" | {motion_tag}"

            # Label background for readability
            (lw, lh), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
            )
            label_y = max(y1 - 6, lh + 4)
            cv2.rectangle(
                frame_bgr,
                (x1, label_y - lh - 2),
                (x1 + lw + 4, label_y + baseline),
                color,
                -1,  # filled
            )
            cv2.putText(
                frame_bgr,
                label,
                (x1 + 2, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),  # white text on coloured background
                1,
                cv2.LINE_AA,
            )

            # Motion arrow — draw inside bbox from centre toward motion direction
            if obj_flow is not None and is_moving:
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                # Scale arrow: cap at half the bbox size so it stays inside
                half_w = max((x2 - x1) // 2 - 4, 8)
                half_h = max((y2 - y1) // 2 - 4, 8)
                mag = obj_flow["object_magnitude"]
                if mag > 0:
                    scale = min(half_w / mag, half_h / mag, 6.0)
                    ex = int(cx + obj_flow["object_dx"] * scale)
                    ey = int(cy + obj_flow["object_dy"] * scale)
                    cv2.arrowedLine(
                        frame_bgr,
                        (cx, cy),
                        (ex, ey),
                        (255, 255, 255),  # white arrow
                        2,
                        tipLength=0.35,
                    )

        # ── Scene-level HUD ──────────────────────────────────────────────────
        hud_y = 30
        if flow_stats:
            cv2.putText(
                frame_bgr,
                f"Scene Flow: {flow_stats['mean_magnitude']:.2f} px/f",
                (10, hud_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 200, 0),
                2,
                cv2.LINE_AA,
            )
            hud_y += 30

        cv2.putText(
            frame_bgr,
            f"Scene Risk: {scene_risk}",
            (10, hud_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 60, 255) if scene_risk > 0 else (60, 200, 60),
            2,
            cv2.LINE_AA,
        )
        hud_y += 30

        if lane_result is not None:
            cv2.putText(
                frame_bgr,
                f"Lane: {lane_result['lane_pixel_ratio'] * 100:.1f}%  "
                f"Drivable: {lane_result['drivable_pixel_ratio'] * 100:.1f}%",
                (10, hud_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (64, 64, 255),
                2,
                cv2.LINE_AA,
            )

        return frame_bgr