from core.detection.yolo_detector import YOLODetector
from core.depth.stereo_depth import StereoDepth
from core.optical_flow.global_flow import GlobalOpticalFlow
from core.optical_flow.object_flow import ObjectOpticalFlow
from core.lane.lane_detector import LaneDetector
from core.calculation.risk_engine import RiskEngine
from config.settings import LANE_CONFIG
from config.settings import ANNOTATION_CONFIG

from database.repository import (
    FrameRepository,
    DetectionRepository,
    OpticalFlowRepository,
    SceneRepository,
    LaneRepository,
    PerformanceRepository,
)

from utils.logger import get_logger

import cv2
import numpy as np
import time


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
        self.perf_repo = PerformanceRepository()

    def reset(self):
        self.flow.reset()
        self.logger.info("Pipeline state reset.")

    def process_frame(self, frame_data, frame_saver=None):

        frame_id = frame_data.frame_id
        self.logger.info(f"Processing frame {frame_id}")
        pipeline_start = time.perf_counter()

        # 1️⃣ Save frame info
        self.frame_repo.insert(frame_data)

        # 2️⃣ Detection
        t_detect = time.perf_counter()
        detections = self.detector.detect(frame_data.rgb_image)
        yolo_ms = (time.perf_counter() - t_detect) * 1000.0
        self.logger.info(f"[Frame {frame_id}] Detected {len(detections)} objects")

        # 3️⃣ Optical Flow — global scene level
        #    compute() returns (stats_dict, flow_field) or None on the first frame.
        #    flow_field is a fresh (H, W, 2) numpy array for *this* frame pair;
        #    it is never stored on GlobalOpticalFlow — we pass it forward explicitly.
        t_flow = time.perf_counter()
        flow_result = self.flow.compute(frame_data.rgb_image)
        flow_stats: dict | None = None
        flow_field: np.ndarray | None = None

        if flow_result is not None:
            flow_stats, flow_field = flow_result  # unpack tuple

        self.flow_repo.insert(frame_id, flow_stats)
        global_flow_ms = (time.perf_counter() - t_flow) * 1000.0

        if flow_stats is not None:
            self.logger.info(
                f"[Frame {frame_id}] Flow mean={flow_stats['mean_magnitude']:.4f}"
            )
        else:
            self.logger.info(f"[Frame {frame_id}] First frame (no flow)")

        # 4️⃣ Per-object optical flow
        #    ObjectOpticalFlow is stateless: it receives the fresh flow_field
        #    as an argument and slices it per bounding box.  No cached state.
        t_obj_flow = time.perf_counter()
        if flow_field is not None:
            object_flow_list = self.object_flow.compute_object_flows(
                flow_field, detections
            )
        else:
            # First frame — no previous frame to compare against
            object_flow_list = [None] * len(detections)
        object_flow_ms = (time.perf_counter() - t_obj_flow) * 1000.0

        # 5️⃣ Lane Segmentation (needed for path occupancy / drivable capacity)
        lane_result = None
        lane_ms = 0.0
        if self.lane_detector is not None:
            t_lane = time.perf_counter()
            lane_result = self.lane_detector.detect(frame_data.rgb_image)
            self.lane_repo.insert(frame_id, lane_result)
            lane_ms = (time.perf_counter() - t_lane) * 1000.0
            self.logger.info(
                f"[Frame {frame_id}] Lane ratio={lane_result['lane_pixel_ratio']:.4f}, "
                f"Drivable ratio={lane_result['drivable_pixel_ratio']:.4f}"
            )

        # 6️⃣ Per-object: depth → object flow → fused risk
        t_risk = time.perf_counter()
        calculations = []
        for det, obj_flow in zip(detections, object_flow_list):

            distance = self.depth.compute_distance(frame_data.depth_map, det["bbox"])
            object_context = self.risk_engine.assess_object_risk(
                distance,
                object_flow=obj_flow,
                lane_result=lane_result,
                bbox=det["bbox"],
                class_id=det["class_id"],
            )

            risk = object_context["risk"]

            if obj_flow or object_context.get("lane_overlap_ratio", 0.0) > 0:
                self.logger.info(
                    f"[Frame {frame_id}] {det['class_id']} risk={risk} "
                    f"score={object_context['risk_score']:.1f} lane={object_context['lane_overlap_ratio']:.2f} "
                    f"moving={bool(obj_flow and obj_flow.get('is_moving'))}"
                )

            result = {
                "frame_id": frame_id,
                "class_id": det["class_id"],
                "class_name": det.get("class_name", str(det["class_id"])),
                "confidence": det["confidence"],
                "bbox": det["bbox"],
                "distance_m": distance,
                "risk": risk,
                "risk_score": object_context["risk_score"],
                "lane_overlap_ratio": object_context["lane_overlap_ratio"],
                "path_occupancy_risk": object_context["path_occupancy_risk"],
                "proximity_score": object_context["proximity_score"],
                "motion_score": object_context["motion_score"],
                "object_flow": obj_flow,  # None on first frame
            }

            calculations.append(result)
            self.det_repo.insert(result)
        risk_ms = (time.perf_counter() - t_risk) * 1000.0

        # 7️⃣ Scene Fusion Metrics
        t_scene = time.perf_counter()
        scene_metrics = self.risk_engine.compute_scene_metrics(
            calculations,
            flow_stats=flow_stats,
            lane_result=lane_result,
        )
        scene_risk = int(scene_metrics.get("scene_risk_score") or 0)
        alert_flag = int(scene_metrics.get("alert_flag") or 0)

        self.scene_repo.insert(frame_id, scene_metrics)
        scene_ms = (time.perf_counter() - t_scene) * 1000.0
        self.logger.info(
            f"[Frame {frame_id}] Scene risk={scene_risk}, "
            f"path={scene_metrics['path_occupancy_risk']:.1f}, "
            f"hazard={scene_metrics['dynamic_hazard_index']:.1f}, "
            f"capacity={scene_metrics['drivable_capacity_score']:.1f}, "
            f"safety={scene_metrics['trip_safety_score']:.1f}"
        )

        # 8️⃣ Save Annotated Frame
        annotation_ms = 0.0
        pre_annotation_total_ms = (time.perf_counter() - pipeline_start) * 1000.0
        pre_annotation_fps = 1000.0 / pre_annotation_total_ms if pre_annotation_total_ms > 0 else None
        perf_overlay = {
            "pipeline_total_ms": pre_annotation_total_ms,
            "pipeline_fps": pre_annotation_fps,
        }
        if frame_saver is not None:
            self.logger.info(f"[Frame {frame_id}] Saving annotated frame")
            t_annotation = time.perf_counter()
            annotated_frame = self._draw_annotations(
                frame_data.rgb_image,
                calculations,
                flow_stats,
                scene_risk,
                lane_result,
                perf_stats=perf_overlay,
            )
            frame_saver.save(frame_id, annotated_frame)
            annotation_ms = (time.perf_counter() - t_annotation) * 1000.0

        pipeline_total_ms = (time.perf_counter() - pipeline_start) * 1000.0
        pipeline_fps = 1000.0 / pipeline_total_ms if pipeline_total_ms > 0 else None

        perf_stats = {
            "yolo_ms": yolo_ms,
            "global_flow_ms": global_flow_ms,
            "object_flow_ms": object_flow_ms,
            "lane_ms": lane_ms,
            "risk_ms": risk_ms,
            "scene_ms": scene_ms,
            "annotation_ms": annotation_ms,
            "pipeline_total_ms": pipeline_total_ms,
            "pipeline_fps": pipeline_fps,
            "detection_count": len(detections),
        }
        self.perf_repo.insert(frame_id, perf_stats)
        self.logger.info(
            f"[Frame {frame_id}] Perf total={pipeline_total_ms:.2f}ms "
            f"({pipeline_fps:.2f} FPS) | YOLO={yolo_ms:.2f}ms, "
            f"Flow={global_flow_ms:.2f}ms, ObjFlow={object_flow_ms:.2f}ms, "
            f"Lane={lane_ms:.2f}ms, Risk={risk_ms:.2f}ms, Scene={scene_ms:.2f}ms"
        )

        return {
            "frame_id": frame_id,
            "detections": calculations,
            "flow": flow_stats,
            "lane": lane_result,
            "scene_metrics": scene_metrics,
            "scene_risk": scene_risk,
            "alert": alert_flag == 1,
            "performance": perf_stats,
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
        perf_stats: dict | None = None,
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
            bbox_thickness = int(ANNOTATION_CONFIG.get("bbox_thickness", 2))

            # Bounding box
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, bbox_thickness)

            # Distance text
            distance = det.get("distance_m")
            dist_text = f"{distance:.2f}m" if distance is not None else "N/A"

            # MOV / STA + magnitude badge
            if obj_flow is not None:
                motion_tag = f"MOV {obj_flow['object_magnitude']:.1f}px" if is_moving else "STA"
            else:
                motion_tag = ""

            class_label = det.get("class_name", str(det.get("class_id", "obj")))
            label = f"{class_label} | {dist_text} | {det['risk']}"
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
            hud_y += 25

        if perf_stats is not None:
            total_ms = perf_stats.get("pipeline_total_ms")
            fps = perf_stats.get("pipeline_fps")
            if total_ms is not None and fps is not None:
                cv2.putText(
                    frame_bgr,
                    f"Inference: {total_ms:.1f} ms | FPS: {fps:.1f}",
                    (10, hud_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        return frame_bgr
