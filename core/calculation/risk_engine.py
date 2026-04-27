from typing import Dict, Optional, Any

import numpy as np

from config.settings import OBJECT_FLOW_CONFIG, RISK_CONFIG, RISK_FUSION_CONFIG


class RiskEngine:

    @staticmethod
    def _clamp(value: float, minimum: float = 0.0, maximum: float = 100.0) -> float:
        return max(minimum, min(maximum, float(value)))

    @staticmethod
    def _lane_overlap_ratio(bbox, lane_result: Optional[Dict]) -> float:
        if not lane_result:
            return 0.0

        da_mask = lane_result.get("da_mask")
        if da_mask is None:
            return 0.0

        x1, y1, x2, y2 = map(int, bbox)
        h, w = da_mask.shape[:2]
        x1 = max(0, min(w, x1))
        y1 = max(0, min(h, y1))
        x2 = max(0, min(w, x2))
        y2 = max(0, min(h, y2))

        if x2 <= x1 or y2 <= y1:
            return 0.0

        roi = da_mask[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0

        return float(np.count_nonzero(roi == 1) / roi.size)

    @staticmethod
    def _proximity_score(distance_m: float) -> float:
        if distance_m <= RISK_CONFIG["high_distance_m"]:
            return 100.0
        if distance_m <= RISK_CONFIG["medium_distance_m"]:
            return 60.0
        return max(0.0, 60.0 - (distance_m - RISK_CONFIG["medium_distance_m"]) * 4.0)

    @staticmethod
    def _motion_score(object_flow: Optional[Dict]) -> float:
        if not object_flow:
            return 0.0

        magnitude = float(object_flow.get("object_magnitude", 0.0) or 0.0)
        threshold = float(OBJECT_FLOW_CONFIG["moving_threshold"])
        if threshold <= 0:
            return 0.0

        raw_score = (magnitude / threshold) * 60.0
        if not object_flow.get("is_moving"):
            raw_score *= 0.35

        return RiskEngine._clamp(raw_score)

    @staticmethod
    def _class_weight(class_id: Optional[int]) -> float:
        if class_id is None:
            return 1.0

        class_weights = RISK_FUSION_CONFIG.get("class_weights", {})
        try:
            return float(class_weights.get(int(class_id), 1.0))
        except Exception:
            return 1.0

    @staticmethod
    def _flow_score(flow_stats: Optional[Dict[str, float]]) -> float:
        if not flow_stats:
            return 0.0

        mean_magnitude = float(flow_stats.get("mean_magnitude", 0.0) or 0.0)
        normalizer = float(RISK_FUSION_CONFIG.get("flow_normalizer", 6.0))
        if normalizer <= 0:
            return 0.0

        return RiskEngine._clamp((mean_magnitude / normalizer) * 100.0)

    def assess_object_risk(
        self,
        distance_m: Optional[float],
        object_flow: Optional[Dict] = None,
        lane_result: Optional[Dict] = None,
        bbox: Optional[Any] = None,
        class_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        lane_overlap_ratio = self._lane_overlap_ratio(bbox, lane_result) if bbox is not None else 0.0
        lane_score = lane_overlap_ratio * 100.0
        class_weight = self._class_weight(class_id)

        if distance_m is None:
            motion_score = self._motion_score(object_flow)
            contextual_score = (
                0.60 * motion_score
                + 0.40 * lane_score
            ) * class_weight
            risk_score = self._clamp(contextual_score)

            if risk_score >= RISK_FUSION_CONFIG["object_high_threshold"]:
                risk = "HIGH"
            elif risk_score >= RISK_FUSION_CONFIG["object_medium_threshold"]:
                risk = "MEDIUM"
            elif motion_score > 0.0 or lane_overlap_ratio > 0.0:
                risk = "LOW"
            else:
                risk = "UNKNOWN"

            return {
                "risk": risk,
                "risk_score": risk_score,
                "proximity_score": 0.0,
                "motion_score": motion_score,
                "lane_overlap_ratio": lane_overlap_ratio,
                "path_occupancy_risk": self._clamp(risk_score * lane_overlap_ratio),
                "class_weight": class_weight,
            }

        proximity_score = self._proximity_score(distance_m)
        motion_score = self._motion_score(object_flow)

        base_score = (
            RISK_FUSION_CONFIG["proximity_weight"] * proximity_score
            + RISK_FUSION_CONFIG["motion_weight"] * motion_score
            + RISK_FUSION_CONFIG["lane_weight"] * lane_score
        )

        risk_score = self._clamp(base_score * class_weight)

        if object_flow and object_flow.get("is_moving"):
            risk_score = self._clamp(
                risk_score + (lane_overlap_ratio * RISK_FUSION_CONFIG["moving_object_bias"])
            )

        if risk_score >= RISK_FUSION_CONFIG["object_high_threshold"]:
            risk = "HIGH"
        elif risk_score >= RISK_FUSION_CONFIG["object_medium_threshold"]:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        path_occupancy_risk = self._clamp(risk_score * lane_overlap_ratio)

        return {
            "risk": risk,
            "risk_score": risk_score,
            "proximity_score": proximity_score,
            "motion_score": motion_score,
            "lane_overlap_ratio": lane_overlap_ratio,
            "path_occupancy_risk": path_occupancy_risk,
            "class_weight": class_weight,
        }

    def estimate_risk(
        self,
        distance_m: Optional[float],
        object_flow: Optional[Dict] = None,
        lane_result: Optional[Dict] = None,
        bbox: Optional[Any] = None,
        class_id: Optional[int] = None,
    ) -> str:
        """
        Estimate risk level for a single detected object.

        Args:
            distance_m:   Distance to the object in metres (None = unknown).
            object_flow:  Per-object flow dict from ObjectOpticalFlow, or None
                          when optical flow is not available (e.g. first frame).

        Returns:
            One of "HIGH" / "MEDIUM" / "LOW" / "UNKNOWN".

        The score is no longer a simple distance-only downgrade. It fuses:
            - proximity to the ego camera
            - object motion magnitude and direction
            - object overlap with the drivable path
            - class weighting for vulnerable / heavy road users
        """
        return self.assess_object_risk(
            distance_m,
            object_flow=object_flow,
            lane_result=lane_result,
            bbox=bbox,
            class_id=class_id,
        )["risk"]

    def compute_scene_risk(self, object_calcs: list) -> int:
        """Return count of objects at HIGH risk in this frame."""
        return sum(1 for c in object_calcs if c["risk"] == "HIGH")

    def compute_scene_metrics(
        self,
        object_calcs: list,
        flow_stats: Optional[Dict[str, float]] = None,
        lane_result: Optional[Dict] = None,
    ) -> Dict[str, float | int | None]:
        scene_risk_score = self.compute_scene_risk(object_calcs)

        object_risk_scores = [float(c.get("risk_score", 0.0) or 0.0) for c in object_calcs]
        path_occupancy_risk = sum(float(c.get("path_occupancy_risk", 0.0) or 0.0) for c in object_calcs)
        path_occupancy_risk = self._clamp(path_occupancy_risk)

        avg_object_risk = float(np.mean(object_risk_scores)) if object_risk_scores else 0.0
        moving_objects = sum(1 for c in object_calcs if c.get("object_flow") and c["object_flow"].get("is_moving"))
        moving_pressure = min(100.0, moving_objects * RISK_FUSION_CONFIG["moving_object_bias"])
        flow_score = self._flow_score(flow_stats)

        drivable_ratio = float(lane_result.get("drivable_pixel_ratio", 0.0)) if lane_result else 0.0
        drivable_capacity_score = self._clamp((drivable_ratio * 100.0) - path_occupancy_risk)

        dynamic_hazard_index = self._clamp(
            RISK_FUSION_CONFIG["hazard_weight"] * avg_object_risk
            + RISK_FUSION_CONFIG["path_occupancy_weight"] * path_occupancy_risk
            + RISK_FUSION_CONFIG["flow_weight"] * flow_score
            + 0.10 * moving_pressure
        )

        trip_safety_score = self._clamp(
            RISK_FUSION_CONFIG["capacity_weight"] * drivable_capacity_score
            + (1.0 - RISK_FUSION_CONFIG["capacity_weight"]) * (100.0 - dynamic_hazard_index)
        )

        alert_flag = 1 if (scene_risk_score > 0 or dynamic_hazard_index >= 60.0 or path_occupancy_risk >= 35.0) else 0

        return {
            "scene_risk_score": scene_risk_score,
            "path_occupancy_risk": path_occupancy_risk,
            "dynamic_hazard_index": dynamic_hazard_index,
            "drivable_capacity_score": drivable_capacity_score,
            "trip_safety_score": trip_safety_score,
            "alert_flag": alert_flag,
        }
