"""
Simple GPU Profiling Tool - Measure GPU utilization during YOLO inference
Uses pre-extracted frame images instead of BAG file parsing
"""

# MUST be first - set up Python path before any imports
import sys
from pathlib import Path

_script_dir = Path(__file__).parent  # tools/
_parent_dir = _script_dir.parent    # assistance_car/
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

import argparse
import csv
import subprocess
import time
from typing import Dict, List
from glob import glob

import cv2
import torch

from config.settings import YOLO_MODEL_PATH
from core.detection.yolo_detector import YOLODetector
from utils.logger import get_logger 

logger = get_logger(__name__)


def get_gpu_stats() -> Dict[str, float]:
    """
    Query NVIDIA GPU stats using nvidia-smi
    Returns: dict with memory_used_mb, memory_total_mb, utilization_percent, temp_c
    """
    try:
        # Query format: memory.used, memory.total, utilization.gpu, temperature.gpu
        cmd = [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            return {
                "memory_used_mb": 0,
                "memory_total_mb": 0,
                "gpu_util_percent": 0,
                "temp_c": 0,
            }

        parts = result.stdout.strip().split(",")
        return {
            "memory_used_mb": float(parts[0].strip()),
            "memory_total_mb": float(parts[1].strip()),
            "gpu_util_percent": float(parts[2].strip()),
            "temp_c": float(parts[3].strip()),
        }
    except Exception as e:
        logger.warning(f"Failed to query GPU stats: {e}")
        return {
            "memory_used_mb": 0,
            "memory_total_mb": 0,
            "gpu_util_percent": 0,
            "temp_c": 0,
        }


def main():
    parser = argparse.ArgumentParser(description="Simple GPU Profiling Tool for YOLO Inference")
    parser.add_argument(
        "--image-dir", required=True, help="Directory containing frame images (.jpg or .png)"
    )
    parser.add_argument(
        "--model", default=None, help="YOLO model path (default: from config)"
    )
    parser.add_argument(
        "--frames", type=int, default=50, help="Number of frames to profile"
    )
    parser.add_argument(
        "--out", default="gpu_profile_simple.csv", help="Output CSV file"
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("GPU PROFILING TOOL - YOLO Inference Analysis")
    logger.info("=" * 60)

    # Check CUDA availability
    if not torch.cuda.is_available():
        logger.error("CUDA not available! Cannot profile GPU.")
        return 1

    logger.info(f"GPU Device: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA Version: {torch.version.cuda}")

    # Initialize detector with GPU
    model_path = args.model or str(YOLO_MODEL_PATH)
    logger.info(f"Loading YOLO model from {model_path}...")
    detector = YOLODetector(model_path, device="cuda")
    logger.info(f"Detector device: {detector.device}")

    # Load image files
    logger.info(f"Loading frame images from {args.image_dir}...")
    image_patterns = [
        str(Path(args.image_dir) / "**" / "*.jpg"),
        str(Path(args.image_dir) / "**" / "*.png"),
    ]
    image_files = []
    for pattern in image_patterns:
        image_files.extend(sorted(glob(pattern, recursive=True)))
    
    if not image_files:
        logger.error(f"No image files found in {args.image_dir}")
        return 1
    
    logger.info(f"Found {len(image_files)} images, profiling {args.frames}...")

    # Profile results
    profile_results = []

    try:
        for frame_idx in range(min(args.frames, len(image_files))):
            image_path = image_files[frame_idx]
            frame = cv2.imread(image_path)
            
            if frame is None:
                logger.warning(f"Failed to load image: {image_path}")
                continue

            logger.info(f"\n--- Frame {frame_idx}: {Path(image_path).name} ---")

            # Single inference timing
            gpu_before = get_gpu_stats()
            start = time.perf_counter()
            detections = detector.detect(frame)
            elapsed_ms = (time.perf_counter() - start) * 1000
            gpu_after = get_gpu_stats()

            logger.info(f"Inference time: {elapsed_ms:.1f} ms")
            logger.info(f"Detections: {len(detections)}")
            logger.info(
                f"GPU util: {gpu_before['gpu_util_percent']:.1f}% → {gpu_after['gpu_util_percent']:.1f}%"
            )
            logger.info(
                f"GPU memory: {gpu_before['memory_used_mb']:.0f}MB → {gpu_after['memory_used_mb']:.0f}MB"
            )
            logger.info(f"GPU temp: {gpu_before['temp_c']:.1f}°C → {gpu_after['temp_c']:.1f}°C")

            profile_results.append(
                {
                    "frame_id": frame_idx,
                    "image_name": Path(image_path).name,
                    "inference_ms": elapsed_ms,
                    "num_detections": len(detections),
                    "gpu_util_before_percent": gpu_before["gpu_util_percent"],
                    "gpu_util_after_percent": gpu_after["gpu_util_percent"],
                    "gpu_memory_before_mb": gpu_before["memory_used_mb"],
                    "gpu_memory_after_mb": gpu_after["memory_used_mb"],
                    "gpu_temp_before_c": gpu_before["temp_c"],
                    "gpu_temp_after_c": gpu_after["temp_c"],
                }
            )

    except Exception as e:
        logger.error(f"Error during profiling: {e}", exc_info=True)
        return 1

    # Write results
    logger.info(f"\nWriting results to {args.out}...")
    if profile_results:
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=profile_results[0].keys())
            writer.writeheader()
            writer.writerows(profile_results)

    # Summary statistics
    if profile_results:
        times = [r["inference_ms"] for r in profile_results]
        gpu_utils = [r["gpu_util_after_percent"] for r in profile_results]
        gpu_mems = [r["gpu_memory_after_mb"] for r in profile_results]

        logger.info("\n" + "=" * 60)
        logger.info("SUMMARY STATISTICS")
        logger.info("=" * 60)
        logger.info(
            f"Inference time - avg: {sum(times)/len(times):.1f} ms, "
            f"min: {min(times):.1f} ms, max: {max(times):.1f} ms, "
            f"median: {sorted(times)[len(times)//2]:.1f} ms"
        )
        logger.info(
            f"GPU util - avg: {sum(gpu_utils)/len(gpu_utils):.1f}%, "
            f"min: {min(gpu_utils):.1f}%, max: {max(gpu_utils):.1f}%"
        )
        logger.info(
            f"GPU memory - avg: {sum(gpu_mems)/len(gpu_mems):.0f}MB, "
            f"peak: {max(gpu_mems):.0f}MB"
        )
        
        # Key finding: GPU utilization during YOLO
        avg_util = sum(gpu_utils) / len(gpu_utils)
        if avg_util < 30:
            logger.warning(f"⚠️  GPU utilization is LOW ({avg_util:.1f}%) - GPU not fully utilized!")
            logger.warning("   Possible causes: PCIe bandwidth limit, batch size too small, or CPU bottleneck")
        else:
            logger.info(f"✓ GPU utilization is reasonable ({avg_util:.1f}%)")

    logger.info("\nGPU profiling complete!")
    return 0


if __name__ == "__main__":
    exit(main())
