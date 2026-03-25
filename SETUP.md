# Setup Guide - Assistance Car

Panduan lengkap untuk setup Assistance Car di mesin lokal Anda.

## Prerequisites

- **Python 3.10+** (3.10.18 recommended)
- **Conda** (optional, tapi recommended untuk dependency management)
- **Git**
- **FFmpeg** (untuk video processing)

### Install FFmpeg

**macOS:**
```bash
brew install ffmpeg
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install ffmpeg
```

**Windows:**
```bash
choco install ffmpeg
# atau download dari https://ffmpeg.org/download.html
```

## Installation Steps

### Option 1: Using Conda (Recommended)

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/assistance_car.git
cd assistance_car

# Create conda environment
conda create -n assistance-car python=3.10 -y
conda activate assistance-car

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Using venv

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/assistance_car.git
cd assistance_car

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

## Download Pre-trained Models

Model sudah dilengkapi di repository ini sesuai kebutuhan (YOLO, Lane). Jika diperlukan download manual:

1. **YOLO Model** (`yolov11n.pt`):
   - Download dari [Ultralytics](https://github.com/ultralytics/assets/releases)
   - Simpan di: `assets/models/yolo.pt`

2. **Lane Model** (`lane_twinlitenet.pth`):
   - Sudah disediakan di: `assets/models/lane_twinlitenet.pth`

## Running the Application

### Web UI (Recommended)

```bash
# Terminal 1: Start FastAPI backend
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Open browser
# Navigate to http://localhost:8000
```

### CLI (No UI)

```bash
python run_backend.py --source path/to/video.bag --type bag
# atau
python run_backend.py --source path/to/video.mp4 --type mp4
```

## RealSense Setup (For Realtime Processing)

Jika menggunakan Intel RealSense:

1. Install RealSense SDK:
```bash
# macOS
brew install librealsense2

# Linux
sudo apt-get install librealsense2

# Windows
# Download dari https://github.com/IntelRealSense/librealsense/releases
```

2. Connect RealSense camera dan test:
```bash
# Linux/macOS
rs-enumerate-devices

# Windows
# Gunakan Intel RealSense Viewer
```

## Configuration

Edit `config/settings.py` untuk customize:

- **Model paths**: `MODEL_PATH`, `LANE_MODEL_PATH`
- **Processing**: `PROCESSING_CONFIG` (fps, max frames, dll)
- **YOLO**: `YOLO_CONFIG` (confidence threshold)
- **Risk levels**: `RISK_CONFIG` (distance thresholds)
- **RealSense**: `REALSENSE_CONFIG` (resolution, fps, dll)
- **Lane**: `LANE_CONFIG` (enabled, model size, device)

## Troubleshooting

### PyRealsense2 ImportError
```bash
pip install pyrealsense2
# atau untuk conda
conda install -c conda-forge pyrealsense2
```

### CUDA/GPU Support
Untuk menggunakan GPU:
```bash
# Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Update lane config device
# Di config/settings.py, ubah:
# LANE_CONFIG["device"] = "cuda"
```

### Low Performance on RealSense
- Reduce `REALSENSE_CONFIG["target_process_fps"]` di settings
- Enable lane skip dengan set `LANE_CONFIG["enabled"] = False`
- Use smaller lane model: `LANE_CONFIG["model_size"] = "nano"`

## Structure

```
assistance_car/
├── app.py                      # FastAPI Web UI
├── run_backend.py              # CLI entry point
├── config/settings.py          # Configuration
├── core/                       # Pipeline & models
│   ├── pipeline.py             # Main perception pipeline
│   ├── detection/              # YOLO detection
│   ├── depth/                  # Stereo depth
│   ├── optical_flow/           # Motion flow
│   ├── lane/                   # Lane segmentation
│   └── calculation/            # Risk engine
├── services/                   # Data generators & processors
├── database/                   # SQLite persistence
├── utils/                      # Helpers & logging
├── assets/                     # Models & outputs
│   ├── models/                 # Checkpoints
│   └── output/                 # Generated videos
└── logs/                       # Activity logs
```

## Next Steps

1. Download test `.bag` atau `.mp4` file
2. Buka http://localhost:8000 di browser
3. Upload file dan klik Process
4. Tunggu hasil dan download video output

Selamat mencoba! 🚗💨
