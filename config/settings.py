from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

ASSETS_DIR = BASE_DIR / "assets"
OUTPUT_DIR = ASSETS_DIR / "output"
TEMP_FRAMES_DIR = ASSETS_DIR / "temp_frames"
LOG_DIR = BASE_DIR / "logs"

YOLO_MODEL_PATH = BASE_DIR / "assets" / "models" / "yolo_best.pt"
LANE_MODEL_PATH = BASE_DIR / "assets" / "models" / "lane_twinlitenet_best.pth"

DB_PATH = BASE_DIR / "perception.db"
SCHEMA_PATH = BASE_DIR / "database" / "schema.sql"

PROCESSING_CONFIG = {
	"default_fps": 30.0,
	# "max_bag_frames": 2000,
 	"max_bag_frames": 20000,
	"clean_temp_frames_on_start": True,
}

YOLO_CONFIG = {
	"confidence_threshold": 0.25,
	"input_resize_enabled": True,
	"input_width": 416,
	"input_height": 320,
	"fp16_enabled": True,
}

RISK_CONFIG = {
	"high_distance_m": 5.0,
	"medium_distance_m": 10.0,
	"min_valid_depth_pixels": 10,
}

RISK_FUSION_CONFIG = {
	"object_high_threshold": 80.0,
	"object_medium_threshold": 60.0,
	"proximity_weight": 0.25,
	"motion_weight": 0.10,
	"lane_weight": 0.08,
	"flow_weight": 0.15,
	"path_occupancy_weight": 0.12,
	"hazard_weight": 0.40,
	"capacity_weight": 0.60,
	"moving_object_bias": 6.0,
	"flow_normalizer": 6.0,
	"scene_alert_min_high_risk_objects": 2,
	"scene_alert_hazard_threshold": 70.0,
	"scene_alert_occupancy_threshold": 50.0,
	"class_weights": {
		0: 1.10,
		1: 1.08,
		2: 1.00,
		3: 1.06,
		5: 1.05,
		7: 1.05,
	},
}

FLOW_CONFIG = {
	"pyr_scale": 0.5,
	"levels": 3,
	"winsize": 15,
	"iterations": 2,
	"poly_n": 5,
	"poly_sigma": 1.2,
	"flags": 0,
}

OBJECT_FLOW_CONFIG = {
	# Object is considered "moving" if its bbox mean flow magnitude exceeds this (px/frame)
	"moving_threshold": 1.5,
}

VIDEO_CONFIG = {
	"default_fps": 20.0,
	"codec": "mp4v",
	"ffmpeg_command": "ffmpeg",
	"audio_codec": "aac",
	"audio_bitrate": "192k",
	"keep_intermediate_silent_video": False,
}

AUDIO_ALERT_CONFIG = {
	"sample_rate": 44100,
	"beep_hz": 1200,
	"pulses_per_sec": 3,
	"duty_cycle": 0.35,
	"volume": 0.35,
}

UI_CONFIG = {
	"title": "Assistance Car Perception System",
	"max_upload_size_note": "Sementara gunakan path lokal BAG yang valid (opsi MP4 sedang dinonaktifkan).",
}

ANNOTATION_CONFIG = {
	"bbox_thickness": 0.5,
}

REALSENSE_CONFIG = {
	"width": 640,
	"height": 480,
	"fps": 30,
	"target_process_fps": 12,
	"frame_timeout_ms": 5000,
	"jpeg_quality": 80,
}

LANE_CONFIG = {
	"enabled": True,
	"model_size": "small",
	"input_h": 288,
	"input_w": 512,
	"process_every_n_frames": 2,
	"reuse_previous_frame": True,
	"device": "cuda",
}

FLOW_OPTIMIZATION_CONFIG = {
	"batch_enabled": True,
	"batch_size": 2,
	"resolution_reduction_enabled": True,
	"resolution_scale": 0.5,  # Compute flow at 50% resolution (25% memory/compute)
	"skip_enabled": True,
	"skip_every_n_frames": 2,  # Compute flow every 2 frames, reuse for in-between
	"reuse_previous_flow": True,
}

GUIDANCE_CONFIG = {
	"enabled": True,
	"default_mode": "information",  # "information" or "driving"
	"show_technical_metrics": True,
	"show_guidance_metrics": True,
	# Thresholds for safety mapping
	"safety_critical_threshold": 20.0,
	"safety_danger_threshold": 40.0,
	"safety_caution_threshold": 70.0,
	# Capacity thresholds (%)
	"capacity_blocked_threshold": 8.0,
	"capacity_narrow_threshold": 25.0,
	"capacity_adequate_threshold": 50.0,
	# Occupancy thresholds
	"occupancy_clear_threshold": 25.0,
	"occupancy_moderate_threshold": 40.0,
	"occupancy_high_threshold": 60.0,
	# Traffic density thresholds (detection count)
	"traffic_empty_threshold": 1,
	"traffic_normal_threshold": 4,
	"traffic_heavy_threshold": 7,
	# System health thresholds (FPS)
	"system_laggy_threshold": 1.0,
	"system_adequate_threshold": 3.0,
	"system_smooth_threshold": 5.0,
}