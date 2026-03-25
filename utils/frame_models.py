from dataclasses import dataclass, field
import numpy as np
from typing import List, Dict, Any, Optional


@dataclass
class FrameData:
    frame_id: int
    timestamp: float
    rgb_image: np.ndarray
    depth_map: np.ndarray
    camera_matrix: np.ndarray

    image_path: Optional[str] = None
    depth_path: Optional[str] = None

    detections: List[Dict[str, Any]] = field(default_factory=list)
    flow_stats: Optional[Dict[str, float]] = None
    scene_risk: Optional[float] = None