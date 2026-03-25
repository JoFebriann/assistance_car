import cv2
import numpy as np
from typing import Optional, Dict
from config.settings import FLOW_CONFIG


class GlobalOpticalFlow:
    """
    Global (scene-level) dense optical flow
    using Farneback method.
    """

    def __init__(self):
        self.prev_gray: Optional[np.ndarray] = None

    def reset(self):
        """Reset state (useful when processing new video)."""
        self.prev_gray = None

    def compute(self, rgb_image: np.ndarray) -> Optional[Dict]:
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            return None

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray,
            gray,
            None, # type: ignore
            pyr_scale=FLOW_CONFIG["pyr_scale"],
            levels=FLOW_CONFIG["levels"],
            winsize=FLOW_CONFIG["winsize"],
            iterations=FLOW_CONFIG["iterations"],
            poly_n=FLOW_CONFIG["poly_n"],
            poly_sigma=FLOW_CONFIG["poly_sigma"],
            flags=FLOW_CONFIG["flags"]
        ) # type: ignore

        self.prev_gray = gray

        dx = flow[..., 0]
        dy = flow[..., 1]
        magnitude = np.sqrt(dx ** 2 + dy ** 2)

        return {
            "mean_magnitude": float(np.mean(magnitude)),
            "median_magnitude": float(np.median(magnitude)),
            "std_magnitude": float(np.std(magnitude)),
            "mean_dx": float(np.mean(dx)),
            "mean_dy": float(np.mean(dy))
        }