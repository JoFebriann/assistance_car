from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

ASSETS_DIR = BASE_DIR / "assets"
OUTPUT_DIR = ASSETS_DIR / "output"
TEMP_FRAMES_DIR = ASSETS_DIR / "temp_frames"
LOG_DIR = BASE_DIR / "logs"

MODEL_PATH = BASE_DIR / "assets" / "models" / "yolo.pt"
LANE_MODEL_PATH = BASE_DIR / "assets" / "models" / "lane_twinlitenet_best.pth"

DB_PATH = BASE_DIR / "perception.db"
SCHEMA_PATH = BASE_DIR / "database" / "schema.sql"

PROCESSING_CONFIG = {
	"default_fps": 30.0,
	"max_bag_frames": 2000,
	"clean_temp_frames_on_start": True,
}

YOLO_CONFIG = {
	"confidence_threshold": 0.25,
}

RISK_CONFIG = {
	"high_distance_m": 5.0,
	"medium_distance_m": 10.0,
	"min_valid_depth_pixels": 10,
}

FLOW_CONFIG = {
	"pyr_scale": 0.5,
	"levels": 3,
	"winsize": 15,
	"iterations": 3,
	"poly_n": 5,
	"poly_sigma": 1.2,
	"flags": 0,
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
	"max_upload_size_note": "Gunakan file MP4 atau path lokal BAG yang valid.",
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
	"input_h": 360,
	"input_w": 640,
	"device": "cpu",
}