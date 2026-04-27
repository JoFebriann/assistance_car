from .db import get_connection


def _migrate_detections_schema():
    """
    Add object-flow columns to an existing detections table that predates
    this feature.  SQLite's ALTER TABLE only supports ADD COLUMN, so we
    run each statement idempotently inside a try/except.
    """
    new_cols = [
        ("object_flow_magnitude", "REAL"),
        ("object_flow_dx",        "REAL"),
        ("object_flow_dy",        "REAL"),
        ("is_moving",             "INTEGER"),
        ("risk_score",             "REAL"),
        ("lane_overlap_ratio",     "REAL"),
    ]
    conn = get_connection()
    cur = conn.cursor()
    for col, col_type in new_cols:
        try:
            cur.execute(f"ALTER TABLE detections ADD COLUMN {col} {col_type}")
        except Exception:
            pass  # Column already exists — safe to ignore
    conn.commit()
    conn.close()


def _migrate_scene_metrics_schema():
    """
    Add fused risk columns to scene_metrics for existing databases.
    """
    new_cols = [
        ("path_occupancy_risk", "REAL"),
        ("dynamic_hazard_index", "REAL"),
        ("drivable_capacity_score", "REAL"),
        ("trip_safety_score", "REAL"),
    ]
    conn = get_connection()
    cur = conn.cursor()
    for col, col_type in new_cols:
        try:
            cur.execute(f"ALTER TABLE scene_metrics ADD COLUMN {col} {col_type}")
        except Exception:
            pass
    conn.commit()
    conn.close()


# Run migration once at import time so the DB is always up-to-date.
try:
    _migrate_detections_schema()
    _migrate_scene_metrics_schema()
except Exception:
    pass  # DB may not exist yet during fresh startup; init_database() handles that

class FrameRepository:

    def insert(self, frame_data):
        conn = get_connection()
        cur = conn.cursor()

        cur.execute("""
        INSERT OR REPLACE INTO frames (
            frame_id, timestamp,
            image_path, depth_path,
            fx, fy, cx, cy
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            frame_data.frame_id,
            frame_data.timestamp,
            frame_data.image_path,
            frame_data.depth_path,
            frame_data.camera_matrix[0][0],
            frame_data.camera_matrix[1][1],
            frame_data.camera_matrix[0][2],
            frame_data.camera_matrix[1][2],
        ))

        conn.commit()
        conn.close()

class DetectionRepository:

    def insert(self, det):
        conn = get_connection()
        cur = conn.cursor()

        x1, y1, x2, y2 = det["bbox"]

        # Per-object flow data (may be None when flow is unavailable)
        obj_flow = det.get("object_flow")  # type: ignore[assignment]
        flow_mag = obj_flow["object_magnitude"] if obj_flow else None
        flow_dx  = obj_flow["object_dx"]        if obj_flow else None
        flow_dy  = obj_flow["object_dy"]        if obj_flow else None
        is_moving = (1 if obj_flow["is_moving"] else 0) if obj_flow else None
        risk_score = det.get("risk_score")
        lane_overlap_ratio = det.get("lane_overlap_ratio")

        cur.execute("""
        INSERT INTO detections (
            frame_id, class_id, confidence,
            bbox_x1, bbox_y1, bbox_x2, bbox_y2,
            distance_m, risk_level, risk_score, lane_overlap_ratio,
            object_flow_magnitude, object_flow_dx, object_flow_dy, is_moving
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            det["frame_id"],
            det["class_id"],
            det["confidence"],
            x1, y1, x2, y2,
            det["distance_m"],
            det["risk"],
            risk_score,
            lane_overlap_ratio,
            flow_mag, flow_dx, flow_dy, is_moving,
        ))

        conn.commit()
        conn.close()

class SceneRepository:

    def insert(self, frame_id, scene_metrics):
        conn = get_connection()
        cur = conn.cursor()

        cur.execute("""
        INSERT OR REPLACE INTO scene_metrics (
            frame_id,
            scene_risk_score,
            path_occupancy_risk,
            dynamic_hazard_index,
            drivable_capacity_score,
            trip_safety_score,
            alert_flag
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            frame_id,
            scene_metrics.get("scene_risk_score"),
            scene_metrics.get("path_occupancy_risk"),
            scene_metrics.get("dynamic_hazard_index"),
            scene_metrics.get("drivable_capacity_score"),
            scene_metrics.get("trip_safety_score"),
            scene_metrics.get("alert_flag"),
        ))

        conn.commit()
        conn.close()

class OpticalFlowRepository:

    def insert(self, frame_id: int, flow_stats: dict | None):
        if flow_stats is None:
            return

        conn = get_connection()
        cur = conn.cursor()

        cur.execute("""
        INSERT OR REPLACE INTO optical_flow (
            frame_id,
            mean_magnitude,
            median_magnitude,
            std_magnitude,
            mean_dx,
            mean_dy
        ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            frame_id,
            flow_stats["mean_magnitude"],
            flow_stats["median_magnitude"],
            flow_stats["std_magnitude"],
            flow_stats["mean_dx"],
            flow_stats["mean_dy"]
        ))

        conn.commit()
        conn.close()


class LaneRepository:

    def insert(self, frame_id: int, lane_stats: dict):
        conn = get_connection()
        cur = conn.cursor()

        cur.execute("""
        INSERT OR REPLACE INTO lane_metrics (
            frame_id,
            lane_pixel_ratio,
            drivable_pixel_ratio
        ) VALUES (?, ?, ?)
        """, (
            frame_id,
            lane_stats.get("lane_pixel_ratio", 0.0),
            lane_stats.get("drivable_pixel_ratio", 0.0)
        ))

        conn.commit()
        conn.close()


class PerformanceRepository:

    def insert(self, frame_id: int, perf: dict):
        conn = get_connection()
        cur = conn.cursor()

        cur.execute(
            """
            INSERT OR REPLACE INTO performance_metrics (
                frame_id,
                yolo_ms,
                global_flow_ms,
                object_flow_ms,
                lane_ms,
                risk_ms,
                scene_ms,
                annotation_ms,
                pipeline_total_ms,
                pipeline_fps,
                detection_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                frame_id,
                perf.get("yolo_ms"),
                perf.get("global_flow_ms"),
                perf.get("object_flow_ms"),
                perf.get("lane_ms"),
                perf.get("risk_ms"),
                perf.get("scene_ms"),
                perf.get("annotation_ms"),
                perf.get("pipeline_total_ms"),
                perf.get("pipeline_fps"),
                perf.get("detection_count"),
            ),
        )

        conn.commit()
        conn.close()

class AnalyticsRepository:

    @staticmethod
    def _percentile(values: list[float], p: float) -> float | None:
        if not values:
            return None

        ordered = sorted(values)
        if len(ordered) == 1:
            return float(ordered[0])

        rank = (len(ordered) - 1) * (p / 100.0)
        low = int(rank)
        high = min(low + 1, len(ordered) - 1)
        frac = rank - low
        return float(ordered[low] * (1.0 - frac) + ordered[high] * frac)

    def summary(self):

        conn = get_connection()
        cur = conn.cursor()

        cur.execute("SELECT COUNT(*) FROM frames")
        total_frames = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM detections")
        total_detections = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM detections WHERE risk_level='HIGH'")
        high_risk = cur.fetchone()[0]

        cur.execute("SELECT AVG(risk_score) FROM detections")
        avg_object_risk_score = cur.fetchone()[0]

        cur.execute("SELECT AVG(lane_overlap_ratio) FROM detections")
        avg_lane_overlap_ratio = cur.fetchone()[0]

        cur.execute("SELECT AVG(mean_magnitude) FROM optical_flow")
        avg_flow = cur.fetchone()[0]

        cur.execute("SELECT AVG(scene_risk_score) FROM scene_metrics")
        avg_scene_risk_score = cur.fetchone()[0]

        cur.execute("SELECT AVG(path_occupancy_risk) FROM scene_metrics")
        avg_path_occupancy_risk = cur.fetchone()[0]

        cur.execute("SELECT AVG(dynamic_hazard_index) FROM scene_metrics")
        avg_dynamic_hazard_index = cur.fetchone()[0]

        cur.execute("SELECT AVG(drivable_capacity_score) FROM scene_metrics")
        avg_drivable_capacity_score = cur.fetchone()[0]

        cur.execute("SELECT AVG(trip_safety_score) FROM scene_metrics")
        avg_trip_safety_score = cur.fetchone()[0]

        cur.execute("SELECT AVG(lane_pixel_ratio) FROM lane_metrics")
        avg_lane_ratio = cur.fetchone()[0]

        cur.execute("SELECT AVG(drivable_pixel_ratio) FROM lane_metrics")
        avg_drivable_ratio = cur.fetchone()[0]

        cur.execute(
            """
            SELECT
                AVG(yolo_ms),
                AVG(global_flow_ms),
                AVG(object_flow_ms),
                AVG(lane_ms),
                AVG(risk_ms),
                AVG(scene_ms),
                AVG(annotation_ms),
                AVG(pipeline_total_ms),
                AVG(pipeline_fps),
                MIN(pipeline_total_ms),
                MAX(pipeline_total_ms),
                AVG(detection_count)
            FROM performance_metrics
            """
        )
        perf_row = cur.fetchone()

        cur.execute("SELECT pipeline_total_ms FROM performance_metrics WHERE pipeline_total_ms IS NOT NULL")
        total_latency_values = [float(r[0]) for r in cur.fetchall()]
        p50_total_ms = self._percentile(total_latency_values, 50)
        p95_total_ms = self._percentile(total_latency_values, 95)

        conn.close()

        return {
            "total_frames": total_frames,
            "total_detections": total_detections,
            "high_risk_objects": high_risk,
            "avg_object_risk_score": avg_object_risk_score,
            "avg_lane_overlap_ratio": avg_lane_overlap_ratio,
            "avg_flow": avg_flow,
            "avg_scene_risk_score": avg_scene_risk_score,
            "avg_path_occupancy_risk": avg_path_occupancy_risk,
            "avg_dynamic_hazard_index": avg_dynamic_hazard_index,
            "avg_drivable_capacity_score": avg_drivable_capacity_score,
            "avg_trip_safety_score": avg_trip_safety_score,
            "avg_lane_ratio": avg_lane_ratio,
            "avg_drivable_ratio": avg_drivable_ratio,
            "avg_yolo_ms": perf_row[0] if perf_row else None,
            "avg_global_flow_ms": perf_row[1] if perf_row else None,
            "avg_object_flow_ms": perf_row[2] if perf_row else None,
            "avg_lane_ms": perf_row[3] if perf_row else None,
            "avg_risk_ms": perf_row[4] if perf_row else None,
            "avg_scene_ms": perf_row[5] if perf_row else None,
            "avg_annotation_ms": perf_row[6] if perf_row else None,
            "avg_pipeline_total_ms": perf_row[7] if perf_row else None,
            "avg_pipeline_fps": perf_row[8] if perf_row else None,
            "min_pipeline_total_ms": perf_row[9] if perf_row else None,
            "max_pipeline_total_ms": perf_row[10] if perf_row else None,
            "avg_detection_per_frame": perf_row[11] if perf_row else None,
            "p50_pipeline_total_ms": p50_total_ms,
            "p95_pipeline_total_ms": p95_total_ms,
        }