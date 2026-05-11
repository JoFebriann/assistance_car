"""
GPU Profiling Tool - Measure GPU utilization during YOLO inference
Diagnoses why GPU speedup is lower than expected
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

import torch
from torch.profiler import profile, record_function, ProfilerActivity

from config.settings import YOLO_MODEL_PATH
from core.detection.yolo_detector import YOLODetector
from services.video_generator import VideoFrameGenerator
from services.bag_generator import BagFrameGenerator
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


def profile_yolo_inference(
    detector: YOLODetector, frame, num_iterations: int = 10
) -> Dict:
    """
    Profile a single YOLO inference with GPU metrics
    """
    results = {
        "inference_times_ms": [],
        "gpu_utils": [],
        "gpu_memory": [],
        "temperatures": [],
    }

    # Warmup
    _ = detector.detect(frame)
    time.sleep(0.1)

    # Profile inference
    for i in range(num_iterations):
        # Pre-inference GPU state
        gpu_before = get_gpu_stats()

        # Measure inference time
        start = time.perf_counter()
        detections = detector.detect(frame)
        elapsed = (time.perf_counter() - start) * 1000  # ms

        # Post-inference GPU state
        gpu_after = get_gpu_stats()

        results["inference_times_ms"].append(elapsed)
        results["gpu_utils"].append(gpu_after["gpu_util_percent"])
        results["gpu_memory"].append(gpu_after["memory_used_mb"])
        results["temperatures"].append(gpu_after["temp_c"])

        time.sleep(0.05)  # Brief pause between inferences

    return results


def profile_with_torch_profiler(
    detector: YOLODetector, frame, output_path: str = None
) -> Dict:
    """
    Use PyTorch profiler to analyze kernel-level GPU activity
    """
    logger.info("Running PyTorch profiler on YOLO inference...")

    # Warmup
    _ = detector.detect(frame)

    # Profile with torch profiler
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        on_trace_ready=None,
    ) as prof:
        with record_function("YOLO_inference"):
            for _ in range(5):
                _ = detector.detect(frame)

    # Extract key metrics
    events = prof.key_averages()
    gpu_events = [e for e in events if e.device_type == "CUDA"]

    stats = {
        "total_cuda_time_ms": sum(e.self_cpu_time_total for e in gpu_events) / 1e3,
        "total_cpu_time_ms": sum(
            e.cpu_time_total for e in events if e.device_type == "CPU"
        )
        / 1e3,
        "num_cuda_kernels": len(gpu_events),
        "top_kernels": [
            {"name": e.key, "time_ms": e.self_cpu_time_total / 1e3} for e in gpu_events[:10]
        ],
    }

    if output_path:
        prof.export_chrome_trace(output_path)
        logger.info(f"Profiler trace saved to {output_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="GPU Profiling Tool for YOLO Inference")
    parser.add_argument(
        "--source", required=True, help="Path to BAG or video file for frame extraction"
    )
    parser.add_argument(
        "--type", default="bag", choices=["bag", "mp4"], help="Source type"
    )
    parser.add_argument(
        "--model", default=None, help="YOLO model path (default: from config)"
    )
    parser.add_argument(
        "--frames", type=int, default=10, help="Number of frames to profile"
    )
    parser.add_argument(
        "--out", default="gpu_profile.csv", help="Output CSV file"
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("GPU PROFILING TOOL - YOLO Inference Analysis")
    logger.info("=" * 60)

    # Check CUDA availability
    if not torch.cuda.is_available():
        logger.error("CUDA not available! Cannot profile GPU.")
        sys.exit(1)

    logger.info(f"GPU Device: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA Version: {torch.version.cuda}")

    # Initialize detector with GPU
    model_path = args.model or YOLO_MODEL_PATH
    logger.info(f"Loading YOLO model from {model_path}...")
    detector = YOLODetector(model_path, device="cuda")
    logger.info(f"Detector device: {detector.device}")

    # Generate frames based on type
    logger.info(f"Loading frames from {args.source} ({args.type})...")
    if args.type == "bag":
        frame_gen = BagFrameGenerator(args.source)
    else:  # mp4
        frame_gen = VideoFrameGenerator(args.source)

    # Profile results
    profile_results = []

    try:
        for frame_idx, frame_data in enumerate(frame_gen.generate()):
            if frame_idx >= args.frames:
                break

            logger.info(f"\n--- Frame {frame_idx} ---")

            # Frame generator yields a FrameData object with `rgb_image`,
            # or a dict-like item containing the frame image.
            if hasattr(frame_data, "rgb_image"):
                frame = frame_data.rgb_image
            elif hasattr(frame_data, "frame"):
                frame = frame_data.frame
            else:
                frame = frame_data["frame"]

            # Single inference timing
            gpu_before = get_gpu_stats()
            start = time.perf_counter()
            detections = detector.detect(frame)
            elapsed_ms = (time.perf_counter() - start) * 1000
            gpu_after = get_gpu_stats()

            logger.info(f"Inference time: {elapsed_ms:.1f} ms")
            logger.info(f"Detections: {len(detections)}")
            logger.info(
                f"GPU util before: {gpu_before['gpu_util_percent']:.1f}% → after: {gpu_after['gpu_util_percent']:.1f}%"
            )
            logger.info(
                f"GPU memory: {gpu_before['memory_used_mb']:.0f}MB → {gpu_after['memory_used_mb']:.0f}MB"
            )

            profile_results.append(
                {
                    "frame_id": frame_idx,
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
        sys.exit(1)

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
        logger.info(f"Inference time - avg: {sum(times)/len(times):.1f} ms, "
                   f"min: {min(times):.1f} ms, max: {max(times):.1f} ms")
        logger.info(f"GPU utilization - avg: {sum(gpu_utils)/len(gpu_utils):.1f}%, "
                   f"min: {min(gpu_utils):.1f}%, max: {max(gpu_utils):.1f}%")
        logger.info(f"GPU memory - avg: {sum(gpu_mems)/len(gpu_mems):.0f}MB, "
                   f"peak: {max(gpu_mems):.0f}MB")

    logger.info("GPU profiling complete!")


if __name__ == "__main__":
    main()
