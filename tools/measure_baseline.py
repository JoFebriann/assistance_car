#!/usr/bin/env python3
"""Measure baseline per-frame timings for a short run and export CSV.

Usage:
  python -m tools.measure_baseline --source path/to/file.bag --type bag --max-frames 200 --device auto --out baseline.csv

Notes:
- This script instantiates PerceptionPipeline directly (avoids frame PNG saving and video building).
- It still writes DB entries (same as normal run) so AnalyticsRepository will be populated.
"""
from pathlib import Path
import argparse
import csv
import time

from config.settings import YOLO_MODEL_PATH
from core.pipeline import PerceptionPipeline

# Generators
from services.video_generator import VideoFrameGenerator
from services.bag_generator import BagFrameGenerator

from database.db import reset_database


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--type", choices=["bag", "mp4"], required=True)
    p.add_argument("--max-frames", type=int, default=200)
    p.add_argument("--device", default="auto", help="Detector device: auto/cpu/cuda or cuda:0")
    p.add_argument("--out", default="baseline.csv")
    return p.parse_args()


def main():
    args = parse_args()
    source = args.source
    source_type = args.type
    max_frames = args.max_frames
    device = args.device
    out_path = Path(args.out)

    reset_database()

    pipeline = PerceptionPipeline(str(YOLO_MODEL_PATH), detector_device=device)

    if source_type == "bag":
        gen = BagFrameGenerator(source).generate()
    else:
        gen = VideoFrameGenerator(source).generate()

    rows = []
    frame_count = 0
    start = time.perf_counter()

    for frame in gen:
        if frame_count >= max_frames:
            break
        t0 = time.perf_counter()
        result = pipeline.process_frame(frame, frame_saver=None, annotation_mode="driving")
        t1 = time.perf_counter()

        perf = result.get("performance", {})
        rows.append({
            "frame_id": frame.frame_id,
            "timestamp": frame.timestamp,
            "yolo_ms": perf.get("yolo_ms"),
            "global_flow_ms": perf.get("global_flow_ms"),
            "object_flow_ms": perf.get("object_flow_ms"),
            "lane_ms": perf.get("lane_ms"),
            "risk_ms": perf.get("risk_ms"),
            "scene_ms": perf.get("scene_ms"),
            "annotation_ms": perf.get("annotation_ms"),
            "pipeline_total_ms": perf.get("pipeline_total_ms"),
            "pipeline_fps": perf.get("pipeline_fps"),
            "detection_count": perf.get("detection_count"),
            "wall_time_s": t1 - t0,
        })

        frame_count += 1

    total = time.perf_counter() - start

    # Write CSV
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    print(f"Processed {frame_count} frames in {total:.2f}s — wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
