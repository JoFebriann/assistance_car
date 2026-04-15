from typing import Dict, Optional

from config.settings import RISK_CONFIG


class RiskEngine:

    def estimate_risk(
        self,
        distance_m: Optional[float],
        object_flow: Optional[Dict] = None,
    ) -> str:
        """
        Estimate risk level for a single detected object.

        Args:
            distance_m:   Distance to the object in metres (None = unknown).
            object_flow:  Per-object flow dict from ObjectOpticalFlow, or None
                          when optical flow is not available (e.g. first frame).

        Returns:
            One of "HIGH" / "MEDIUM" / "LOW" / "UNKNOWN".

        Downgrade logic:
            If the object is classified as moving (is_moving=True), its risk
            is reduced by one level:
              HIGH   → MEDIUM  (moving obstacle ahead, lower relative danger)
              MEDIUM → LOW
              LOW    → LOW     (already minimal)
            This reflects the insight that a vehicle moving in the same
            direction as the ego camera poses less immediate danger than a
            stationary object at the same distance.
        """
        # Base risk from distance
        if distance_m is None:
            base = "UNKNOWN"
        elif distance_m < RISK_CONFIG["high_distance_m"]:
            base = "HIGH"
        elif distance_m < RISK_CONFIG["medium_distance_m"]:
            base = "MEDIUM"
        else:
            base = "LOW"

        # Downgrade one level if the object is moving
        if object_flow and object_flow.get("is_moving"):
            if base == "HIGH":
                return "MEDIUM"
            if base == "MEDIUM":
                return "LOW"

        return base

    def compute_scene_risk(self, object_calcs: list) -> int:
        """Return count of objects at HIGH risk in this frame."""
        return sum(1 for c in object_calcs if c["risk"] == "HIGH")