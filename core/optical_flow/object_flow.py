import numpy as np
from typing import Dict, List, Optional
from config.settings import OBJECT_FLOW_CONFIG


class ObjectOpticalFlow:
    """
    Per-object optical flow analysis.

    This class is **stateless** — it receives a fresh dense flow field
    (H, W, 2) from the pipeline on every call and slices it per bounding
    box.  It never stores or reads flow data from previous frames.

    Usage in pipeline:
        result = global_flow.compute(rgb_image)   # (stats, flow_field) or None
        if result is not None:
            stats, flow_field = result
            object_flows = object_optical_flow.compute_object_flows(flow_field, detections)
    """

    def __init__(self):
        self._threshold: float = float(OBJECT_FLOW_CONFIG["moving_threshold"])

    def compute_object_flows(
        self,
        flow_field: np.ndarray,
        detections: List[Dict],
    ) -> List[Optional[Dict]]:
        """
        Compute motion statistics for each detected object.

        Args:
            flow_field: Dense flow array of shape (H, W, 2) with [dx, dy]
                        per pixel.  This is the flow field produced for the
                        *current* frame pair — passed directly from the
                        pipeline, not retrieved from any cached state.
            detections: List of detection dicts, each containing a "bbox"
                        key with (x1, y1, x2, y2) in pixel coordinates.

        Returns:
            A list (same length and order as `detections`) where each
            element is either an object-flow dict or None if the bbox
            region was too small or out of bounds.

        Object-flow dict schema:
            {
                "object_magnitude": float,  # mean px/frame within bbox
                "object_dx":        float,  # mean horizontal motion
                "object_dy":        float,  # mean vertical motion
                "is_moving":        bool,   # True if magnitude > threshold
            }
        """
        h, w = flow_field.shape[:2]
        results: List[Optional[Dict]] = []

        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])

            # Clamp coords to frame boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            if x2 <= x1 or y2 <= y1:
                results.append(None)
                continue

            # Slice the ROI from the flow field
            roi = flow_field[y1:y2, x1:x2]  # shape (roi_h, roi_w, 2)

            dx_roi = roi[..., 0]
            dy_roi = roi[..., 1]
            mag_roi = np.sqrt(dx_roi**2 + dy_roi**2)

            mean_magnitude = float(np.mean(mag_roi))
            mean_dx = float(np.mean(dx_roi))
            mean_dy = float(np.mean(dy_roi))
            is_moving = mean_magnitude > self._threshold

            results.append(
                {
                    "object_magnitude": mean_magnitude,
                    "object_dx": mean_dx,
                    "object_dy": mean_dy,
                    "is_moving": is_moving,
                }
            )

        return results
