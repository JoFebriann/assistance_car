import numpy as np
from config.settings import RISK_CONFIG


class StereoDepth:
    def compute_distance(self, depth_map, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        roi = depth_map[y1:y2, x1:x2]

        if roi.size == 0:
            return None

        valid = roi[np.isfinite(roi) & (roi > 0)]
        if valid.size < RISK_CONFIG["min_valid_depth_pixels"]:
            return None

        return float(np.median(valid))