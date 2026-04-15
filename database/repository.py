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


# Run migration once at import time so the DB is always up-to-date.
try:
    _migrate_detections_schema()
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

        cur.execute("""
        INSERT INTO detections (
            frame_id, class_id, confidence,
            bbox_x1, bbox_y1, bbox_x2, bbox_y2,
            distance_m, risk_level,
            object_flow_magnitude, object_flow_dx, object_flow_dy, is_moving
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            det["frame_id"],
            det["class_id"],
            det["confidence"],
            x1, y1, x2, y2,
            det["distance_m"],
            det["risk"],
            flow_mag, flow_dx, flow_dy, is_moving,
        ))

        conn.commit()
        conn.close()

class SceneRepository:

    def insert(self, frame_id, scene_risk, alert_flag):
        conn = get_connection()
        cur = conn.cursor()

        cur.execute("""
        INSERT OR REPLACE INTO scene_metrics (
            frame_id, scene_risk_score, alert_flag
        ) VALUES (?, ?, ?)
        """, (
            frame_id,
            scene_risk,
            alert_flag
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

class AnalyticsRepository:

    def summary(self):

        conn = get_connection()
        cur = conn.cursor()

        cur.execute("SELECT COUNT(*) FROM frames")
        total_frames = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM detections")
        total_detections = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM detections WHERE risk_level='HIGH'")
        high_risk = cur.fetchone()[0]

        cur.execute("SELECT AVG(mean_magnitude) FROM optical_flow")
        avg_flow = cur.fetchone()[0]

        cur.execute("SELECT AVG(lane_pixel_ratio) FROM lane_metrics")
        avg_lane_ratio = cur.fetchone()[0]

        cur.execute("SELECT AVG(drivable_pixel_ratio) FROM lane_metrics")
        avg_drivable_ratio = cur.fetchone()[0]

        conn.close()

        return {
            "total_frames": total_frames,
            "total_detections": total_detections,
            "high_risk_objects": high_risk,
            "avg_flow": avg_flow,
            "avg_lane_ratio": avg_lane_ratio,
            "avg_drivable_ratio": avg_drivable_ratio,
        }