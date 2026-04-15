-- ===============================
-- FRAMES
-- ===============================
CREATE TABLE IF NOT EXISTS frames (
    frame_id INTEGER PRIMARY KEY,
    timestamp REAL,
    image_path TEXT,
    depth_path TEXT,
    fx REAL,
    fy REAL,
    cx REAL,
    cy REAL
);

-- ===============================
-- DETECTIONS
-- ===============================
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    frame_id INTEGER,
    class_id INTEGER,
    confidence REAL,
    bbox_x1 REAL,
    bbox_y1 REAL,
    bbox_x2 REAL,
    bbox_y2 REAL,
    distance_m REAL,
    risk_level TEXT,
    -- Per-object optical flow (NULL when flow unavailable, e.g. first frame)
    object_flow_magnitude REAL,
    object_flow_dx        REAL,
    object_flow_dy        REAL,
    is_moving             INTEGER,  -- 1 = moving, 0 = stationary, NULL = unknown
    FOREIGN KEY(frame_id) REFERENCES frames(frame_id)
);

-- ===============================
-- OPTICAL FLOW (GLOBAL)
-- ===============================
CREATE TABLE IF NOT EXISTS optical_flow (
    frame_id INTEGER PRIMARY KEY,
    mean_magnitude REAL,
    median_magnitude REAL,
    std_magnitude REAL,
    mean_dx REAL,
    mean_dy REAL,
    FOREIGN KEY(frame_id) REFERENCES frames(frame_id)
);

-- ===============================
-- SCENE LEVEL METRICS
-- ===============================
CREATE TABLE IF NOT EXISTS scene_metrics (
    frame_id INTEGER PRIMARY KEY,
    scene_risk_score REAL,
    alert_flag INTEGER,
    FOREIGN KEY(frame_id) REFERENCES frames(frame_id)
);

-- ===============================
-- LANE METRICS
-- ===============================
CREATE TABLE IF NOT EXISTS lane_metrics (
    frame_id INTEGER PRIMARY KEY,
    lane_pixel_ratio REAL,
    drivable_pixel_ratio REAL,
    FOREIGN KEY(frame_id) REFERENCES frames(frame_id)
);