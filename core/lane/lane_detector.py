from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from config.settings import LANE_CONFIG, LANE_MODEL_PATH
from utils.logger import get_logger

from .twinlitenet_model import TwinLiteNetPlus


class LaneDetector:
    def __init__(self, model_path: str | Path = LANE_MODEL_PATH):
        self.logger = get_logger("LaneDetector")
        self.model_path = Path(model_path)

        preferred_device = str(LANE_CONFIG["device"]).lower()
        if preferred_device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.input_h = int(LANE_CONFIG["input_h"])
        self.input_w = int(LANE_CONFIG["input_w"])

        self.model_size = str(LANE_CONFIG["model_size"])
        checkpoint = torch.load(self.model_path, map_location=self.device)

        state_dict: dict[str, Any]
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            state_dict = checkpoint["model_state"]
            config = checkpoint.get("config", {})
            self.model_size = str(config.get("model_config", self.model_size))
            self.input_h = int(config.get("img_h", self.input_h))
            self.input_w = int(config.get("img_w", self.input_w))
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            raise RuntimeError("Unsupported lane checkpoint format.")

        self.model = TwinLiteNetPlus(model_size=self.model_size)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            self.logger.warning("Lane model missing keys: %s", len(missing))
        if unexpected:
            self.logger.warning("Lane model unexpected keys: %s", len(unexpected))

        self.model.to(self.device)
        self.model.eval()

        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        self.logger.info(
            "Lane model loaded: %s | size=%s | input=%sx%s | device=%s",
            self.model_path,
            self.model_size,
            self.input_w,
            self.input_h,
            self.device,
        )

    def detect(self, rgb_image: np.ndarray) -> dict[str, Any]:
        h_orig, w_orig = rgb_image.shape[:2]
        resized = cv2.resize(rgb_image, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)

        img = torch.from_numpy(resized).to(self.device)
        img = img.permute(2, 0, 1).float().unsqueeze(0) / 255.0
        img = (img - self.mean) / self.std

        with torch.no_grad():
            da_out, ll_out = self.model(img)

        da_mask = da_out.argmax(dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)
        ll_mask = ll_out.argmax(dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)

        da_mask = cv2.resize(da_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        ll_mask = cv2.resize(ll_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

        lane_ratio = float((ll_mask == 1).sum() / ll_mask.size)
        drivable_ratio = float((da_mask == 1).sum() / da_mask.size)

        return {
            "da_mask": da_mask,
            "ll_mask": ll_mask,
            "lane_pixel_ratio": lane_ratio,
            "drivable_pixel_ratio": drivable_ratio,
        }
