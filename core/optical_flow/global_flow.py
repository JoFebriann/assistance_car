import cv2
import numpy as np
from typing import Optional, Dict, Tuple
from config.settings import FLOW_CONFIG, FLOW_OPTIMIZATION_CONFIG


class GlobalOpticalFlow:
    """
    Global (scene-level) dense optical flow using Farneback method.

    compute() returns a tuple (stats_dict, flow_field) so the caller
    can pass the raw flow array (H, W, 2) directly to ObjectOpticalFlow
    without any intermediate caching on this class.
    
    Optimizations:
    - Resolution reduction: compute flow at reduced resolution, scale velocities back
    - Frame-skipping: compute flow every N frames, reuse for intermediate frames
    """

    def __init__(self):
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_gray_reduced: Optional[np.ndarray] = None
        self._last_result: Optional[Tuple[Dict, np.ndarray]] = None
        self._frame_count = 0

    def reset(self):
        """Reset state (useful when processing new video)."""
        self.prev_gray = None
        self.prev_gray_reduced = None
        self._last_result = None
        self._frame_count = 0

    def compute(
        self, rgb_image: np.ndarray, frame_id: Optional[int] = None
    ) -> Optional[Tuple[Dict, np.ndarray]]:
        """
        Compute dense optical flow for the current frame.
        
        With optimizations:
        - Computes at reduced resolution if enabled (scales velocities back to original)
        - Skips computation every N frames and reuses cached result if enabled
        
        Args:
            rgb_image: Input RGB image (H, W, 3)
            frame_id: Optional frame ID for tracking skip intervals
        
        Returns:
            None on the first frame (no previous frame to compare).
            (stats_dict, flow_field) on subsequent frames where
            flow_field has shape (H_original, W_original, 2) — [dx, dy] per pixel
            in original image coordinates.
        """
        # Get optimization config
        use_resolution_reduction = FLOW_OPTIMIZATION_CONFIG.get("resolution_reduction_enabled", False)
        resolution_scale = FLOW_OPTIMIZATION_CONFIG.get("resolution_scale", 0.5)
        use_skip = FLOW_OPTIMIZATION_CONFIG.get("skip_enabled", False)
        skip_every_n = FLOW_OPTIMIZATION_CONFIG.get("skip_every_n_frames", 2)
        reuse_cached = FLOW_OPTIMIZATION_CONFIG.get("reuse_previous_flow", True)
        
        # Track frame count for skipping
        should_compute = True
        if use_skip and frame_id is not None:
            should_compute = (frame_id % skip_every_n == 0)
        elif use_skip and frame_id is None:
            should_compute = (self._frame_count % skip_every_n == 0)
            self._frame_count += 1
        
        # If skipping and we have a cached result, return it
        if not should_compute and reuse_cached and self._last_result is not None:
            return self._last_result
        
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        orig_h, orig_w = gray.shape
        
        # Apply resolution reduction if enabled
        if use_resolution_reduction and resolution_scale < 1.0:
            gray_reduced = cv2.resize(
                gray,
                (int(orig_w * resolution_scale), int(orig_h * resolution_scale)),
                interpolation=cv2.INTER_LINEAR
            )
        else:
            gray_reduced = gray

        if self.prev_gray is None:
            self.prev_gray = gray
            if use_resolution_reduction:
                self.prev_gray_reduced = gray_reduced
            return None

        # Compute flow at reduced resolution
        flow_reduced = cv2.calcOpticalFlowFarneback(
            self.prev_gray_reduced if use_resolution_reduction else self.prev_gray,
            gray_reduced if use_resolution_reduction else gray,
            None,  # type: ignore
            pyr_scale=FLOW_CONFIG["pyr_scale"],
            levels=FLOW_CONFIG["levels"],
            winsize=FLOW_CONFIG["winsize"],
            iterations=FLOW_CONFIG["iterations"],
            poly_n=FLOW_CONFIG["poly_n"],
            poly_sigma=FLOW_CONFIG["poly_sigma"],
            flags=FLOW_CONFIG["flags"],
        )  # type: ignore

        # Scale flow velocities back to original resolution if needed
        if use_resolution_reduction and resolution_scale < 1.0:
            # Resize flow to original size and scale velocities proportionally
            flow = cv2.resize(
                flow_reduced,
                (orig_w, orig_h),
                interpolation=cv2.INTER_LINEAR
            )
            # Scale velocity components back to original resolution
            flow[..., 0] = flow[..., 0] / resolution_scale
            flow[..., 1] = flow[..., 1] / resolution_scale
        else:
            flow = flow_reduced

        self.prev_gray = gray
        if use_resolution_reduction:
            self.prev_gray_reduced = gray_reduced

        dx = flow[..., 0]
        dy = flow[..., 1]
        magnitude = np.sqrt(dx**2 + dy**2)

        stats = {
            "mean_magnitude": float(np.mean(magnitude)),
            "median_magnitude": float(np.median(magnitude)),
            "std_magnitude": float(np.std(magnitude)),
            "mean_dx": float(np.mean(dx)),
            "mean_dy": float(np.mean(dy)),
        }

        # Cache result for potential reuse
        result = (stats, flow)
        self._last_result = result
        
        # Return both stats and the raw flow field (H, W, 2).
        # The pipeline passes flow_field directly to ObjectOpticalFlow;
        # it is never stored on this object.
        return result

    def compute_batch(
        self, rgb_images: list[np.ndarray]
    ) -> list[Optional[Tuple[Dict, np.ndarray]]]:
        """Compute flow for a queued list of frames in order."""
        results: list[Optional[Tuple[Dict, np.ndarray]]] = []
        for rgb_image in rgb_images:
            results.append(self.compute(rgb_image))
        return results
