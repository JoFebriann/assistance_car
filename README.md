# 🚗 Assistance Car Perception System

Sistem persepsi kendaraan berbasis computer vision yang memproses video dari kamera stereo (Intel RealSense `.bag`) atau file video biasa (`.mp4`) untuk melakukan **object detection**, **depth estimation**, **optical flow**, dan **risk assessment** secara real-time per frame. Hasil analisis disimpan ke database SQLite dan ditampilkan melalui antarmuka FastAPI Web UI sederhana (HTML).

---

## Daftar Isi

1. [Gambaran Umum](#1-gambaran-umum)
2. [Arsitektur Sistem](#2-arsitektur-sistem)
3. [Struktur Direktori](#3-struktur-direktori)
4. [Alur Data (Data Flow)](#4-alur-data-data-flow)
5. [Penjelasan Komponen](#5-penjelasan-komponen)
   - [Entry Points](#51-entry-points)
   - [Config](#52-config)
   - [Services](#53-services)
   - [Core — Pipeline](#54-core--pipeline)
   - [Core — Detection](#55-core--detection)
   - [Core — Depth](#56-core--depth)
   - [Core — Optical Flow](#57-core--optical-flow)
   - [Core — Risk Engine](#58-core--risk-engine)
   - [Core — Video Output](#59-core--video-output)
   - [Database](#510-database)
   - [Utils](#511-utils)
6. [Skema Database](#6-skema-database)
7. [Output Artifacts](#7-output-artifacts)
8. [Cara Menjalankan](#8-cara-menjalankan)
9. [Dependensi](#9-dependensi)
10. [Catatan Pengembangan Lanjutan](#10-catatan-pengembangan-lanjutan)

---

## 1. Gambaran Umum

Sistem ini dirancang untuk membantu pengemudi kendaraan dengan menganalisis kondisi jalan secara otomatis. Pengguna memasukkan rekaman dari kamera stereo Intel RealSense (format `.bag`) atau video biasa (`.mp4`), kemudian sistem melakukan:

| Fitur | Deskripsi |
|---|---|
| **Object Detection** | Mendeteksi objek di jalan (kendaraan, pejalan kaki, dll.) menggunakan YOLO |
| **Depth Estimation** | Mengukur jarak objek ke kamera dari data depth stereo (dalam meter) |
| **Optical Flow** | Menganalisis pergerakan global seluruh scene per frame (metode Farnebäck) |
| **Lane + Drivable Segmentation** | Segmentasi lajur dan area drivable menggunakan TwinLiteNetPlus |
| **Risk Assessment** | Menilai level risiko setiap objek (HIGH / MEDIUM / LOW) dan risiko keseluruhan scene |
| **Persistensi Data** | Menyimpan semua hasil analisis ke database SQLite (`perception.db`) |
| **Output Video** | Menghasilkan video anotasi dengan bounding box, jarak, level risiko, dan info optical flow |
| **Audio Alert** | Menghasilkan file WAV dengan bunyi beep saat terdeteksi objek berisiko tinggi |
| **Dashboard** | Menampilkan ringkasan analitik melalui antarmuka FastAPI Web UI |

---

## 2. Arsitektur Sistem

```
┌─────────────────────────────────────────────────────────────┐
│                        ENTRY POINTS                         │
│                                                             │
│    app.py (FastAPI UI)       run_backend.py (CLI)           │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                     SERVICES LAYER                          │
│                                                             │
│                    VideoService                             │
│           (Orchestrator utama per-run)                      │
│                                                             │
│   ┌─────────────────────┐   ┌──────────────────────────┐   │
│   │  BagFrameGenerator  │   │  VideoFrameGenerator     │   │
│   │  (.bag via          │   │  (.mp4 via OpenCV)       │   │
│   │   pyrealsense2)     │   │                          │   │
│   └─────────────────────┘   └──────────────────────────┘   │
└──────────────────────┬──────────────────────────────────────┘
                       │ FrameData stream
┌──────────────────────▼──────────────────────────────────────┐
│                      CORE LAYER                             │
│                                                             │
│                  PerceptionPipeline                         │
│                                                             │
│  ┌──────────────┐  ┌─────────────┐  ┌───────────────────┐  │
│  │ YOLODetector │  │ StereoDepth │  │ GlobalOpticalFlow │  │
│  │  (ultralytics│  │  (median    │  │  (Farnebäck       │  │
│  │   YOLO v11)  │  │   depth ROI)│  │   dense flow)     │  │
│  └──────┬───────┘  └──────┬──────┘  └────────┬──────────┘  │
│         │                 │                   │             │
│         └─────────────────▼───────────────────┘            │
│                        RiskEngine                           │
│             (per-object + scene-level risk)                 │
│                                                             │
│  ┌──────────────┐  ┌─────────────────────────────────────┐  │
│  │  FrameSaver  │  │           VideoBuilder              │  │
│  │  (PNG frames)│  │       + generate_alert_wav          │  │
│  └──────────────┘  └─────────────────────────────────────┘  │
└──────────────────────┬──────────────────────────────────────┘
                       │ INSERT
┌──────────────────────▼──────────────────────────────────────┐
│                    DATABASE LAYER                           │
│                                                             │
│              SQLite — perception.db                         │
│                                                             │
│   frames | detections | optical_flow | scene_metrics        │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Struktur Direktori

```
assistance_car/
│
├── app.py                          # Entry point: Streamlit UI
├── run_backend.py                  # Entry point: CLI (non-GUI)
│
├── config/
│   └── settings.py                 # Konfigurasi path model YOLO
│
├── core/                           # Business logic utama
│   ├── pipeline.py                 # PerceptionPipeline (orkestrasi per-frame)
│   ├── frame_processor.py          # (Placeholder, belum diimplementasi)
│   │
│   ├── detection/
│   │   ├── base_detector.py        # Abstract base class detector
│   │   └── yolo_detector.py        # YOLO v11 via ultralytics
│   │
│   ├── depth/
│   │   └── stereo_depth.py         # StereoDepth: median distance dari depth map
│   │
│   ├── optical_flow/
│   │   └── global_flow.py          # GlobalOpticalFlow: Farnebäck dense flow
│   │
│   ├── lane/                       # Lane segmentation (TwinLiteNetPlus)
│   │
│   ├── calculation/
│   │   └── risk_engine.py          # RiskEngine: per-object & scene risk
│   │
│   └── video/
│       ├── frame_saver.py          # Simpan frame anotasi sebagai PNG
│       ├── video_builder.py        # Rakit PNG menjadi file MP4
│       └── audio_alert.py          # Generate WAV alert (beep saat HIGH risk)
│
├── database/
│   ├── db.py                       # Koneksi SQLite, init & reset database
│   ├── schema.sql                  # DDL: 4 tabel utama
│   └── repository.py               # Repository pattern: 5 kelas repository
│
├── services/
│   ├── video_service.py            # VideoService: orkestrasi level run
│   ├── bag_generator.py            # BagFrameGenerator (pyrealsense2)
│   └── video_generator.py          # VideoFrameGenerator (OpenCV)
│
├── utils/
│   ├── frame_models.py             # FrameData dataclass
│   └── logger.py                   # Logger ke file + console
│
├── assets/
│   ├── models/
│   │   └── yolo.pt                 # Model YOLO yang digunakan
│   ├── temp_frames/                # Frame sementara selama proses (per run_id)
│   ├── output/                     # Output video (.mp4) dan audio (.wav)
│   └── audio/                      # (Reserved)
│
├── logs/
│   └── system.log                  # Log semua aktivitas sistem
│
└── perception.db                   # Database SQLite hasil analisis
```

---

## 4. Alur Data (Data Flow)

Berikut adalah alur data lengkap dari input hingga output:

```
┌─────────────────────┐
│  INPUT              │
│  .bag atau .mp4     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────────────────────┐
│  GENERATOR (per frame)                      │
│                                             │
│  .bag → BagFrameGenerator                  │
│          - Buka file via pyrealsense2        │
│          - Ekstrak: color frame (RGB)        │
│                     depth frame             │
│                     camera intrinsics       │
│                     timestamp               │
│                                             │
│  .mp4 → VideoFrameGenerator                │
│          - Baca frame via OpenCV            │
│          - BGR → RGB                        │
│          - depth_map = zeros (tidak ada)    │
│          - camera_matrix = identity         │
│                                             │
│  Yield: FrameData (tiap frame)              │
└──────────┬──────────────────────────────────┘
           │ FrameData
           ▼
┌─────────────────────────────────────────────┐
│  PerceptionPipeline.process_frame()         │
│                                             │
│  Langkah 1: Simpan info frame               │
│    → FrameRepository.insert()              │
│      → INSERT INTO frames                  │
│                                             │
│  Langkah 2: Object Detection                │
│    → YOLODetector.detect(rgb_image)         │
│      Output: [{bbox, confidence, class_id}] │
│                                             │
│    Untuk tiap deteksi:                      │
│    → StereoDepth.compute_distance(          │
│        depth_map, bbox)                     │
│      Output: distance_m (meter)             │
│      Metode: median pixel dalam ROI bbox    │
│                                             │
│    → RiskEngine.estimate_risk(distance_m)   │
│      < 5m  → HIGH                          │
│      < 10m → MEDIUM                        │
│      ≥ 10m → LOW                           │
│      None  → UNKNOWN                       │
│                                             │
│    → DetectionRepository.insert()          │
│      → INSERT INTO detections              │
│                                             │
│  Langkah 3: Optical Flow                    │
│    → GlobalOpticalFlow.compute(rgb_image)   │
│      Metode: Farnebäck dense flow          │
│      Output: {mean_magnitude,               │
│               median_magnitude,             │
│               std_magnitude,                │
│               mean_dx, mean_dy}             │
│    → OpticalFlowRepository.insert()        │
│      → INSERT INTO optical_flow            │
│                                             │
│  Langkah 4: Scene Risk                      │
│    → RiskEngine.compute_scene_risk()        │
│      = jumlah objek HIGH risk di frame ini  │
│    → SceneRepository.insert()              │
│      → INSERT INTO scene_metrics           │
│                                             │
│  Langkah 5: Simpan Frame Anotasi            │
│    → _draw_annotations() → BGR image        │
│      Anotasi: bounding box berwarna,        │
│               label class + jarak + risk,   │
│               info flow, info scene risk    │
│    → FrameSaver.save() → PNG               │
│      Path: assets/temp_frames/{run_id}/     │
└──────────┬──────────────────────────────────┘
           │ (setelah semua frame selesai)
           ▼
┌─────────────────────────────────────────────┐
│  POST-PROCESSING                            │
│                                             │
│  VideoBuilder.build()                       │
│    - Gabung semua PNG → MP4                 │
│    - FPS: 20                                │
│    - Output: assets/output/{run_id}.mp4     │
│                                             │
│  generate_alert_wav()                       │
│    - 1 sampel audio per frame               │
│    - Beep 1200Hz (3 pulsa/detik) jika alert │
│    - Output: assets/output/{run_id}_alert.wav│
└──────────┬──────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────┐
│  HASIL TERSIMPAN DI perception.db           │
│  + Video & Audio di assets/output/          │
│  + Log di logs/system.log                   │
└─────────────────────────────────────────────┘
```

---

## 5. Penjelasan Komponen

### 5.1 Entry Points

#### `app.py` — FastAPI Web UI

Antarmuka utama berbasis web yang dijalankan dengan `uvicorn app:app --reload`.

Halaman web sederhana menyediakan:
1. Form input source (`bag` atau `mp4`)
2. Trigger proses pipeline end-to-end
3. Ringkasan metrik (Total Frames, Total Detections, High Risk Objects, Average Flow)
4. Preview video output dan audio alert

Alur di `app.py`:
1. `init_database()` dipanggil saat startup FastAPI
2. Saat endpoint `/process` dipanggil → file MP4 disimpan sementara (temporary file) atau path BAG langsung digunakan
3. `VideoService(MODEL_PATH).process(source_path, source_type)` dijalankan
4. Setelah selesai → `AnalyticsRepository().summary()` dipanggil untuk menampilkan ringkasan

#### `run_backend.py` — CLI

Alternatif menjalankan sistem tanpa UI, menggunakan argumen command-line.

```bash
python run_backend.py --source path/to/file.bag --type bag
python run_backend.py --source path/to/file.mp4 --type mp4
```

---

### 5.2 Config

#### `config/settings.py`

```python
BASE_DIR = Path(__file__).resolve().parent.parent  # → assistance_car/
MODEL_PATH = BASE_DIR / "assets" / "models" / "yolo.pt"
```

Satu-satunya konfigurasi global saat ini adalah path ke model YOLO. Semua path lainnya diderivasi dari `BASE_DIR`.

---

### 5.3 Services

#### `services/video_service.py` — `VideoService`

Orkestrator level "per run" — satu call ke `process()` menangani satu file input secara penuh.

Langkah-langkah `process()`:
1. `reset_database()` — Hapus data lama dari 4 tabel
2. `pipeline.reset()` — Reset state optical flow (prev_gray = None)
3. Tentukan `run_id` dari stem nama file input
4. Buat `FrameSaver(run_id)` — siapkan direktori temp frame
5. Pilih generator sesuai `source_type` (`BagFrameGenerator` atau `VideoFrameGenerator`)
6. Iterasi frame: `pipeline.process_frame(frame_data, frame_saver)` per frame
7. `VideoBuilder.build()` — rakit frame PNG jadi video MP4
8. `generate_alert_wav()` — buat audio WAV dari daftar alert flags

#### `services/bag_generator.py` — `BagFrameGenerator`

Membaca file `.bag` dari Intel RealSense menggunakan library `pyrealsense2`.

- Stream yang diaktifkan: `color` dan `depth`
- Mengekstrak **camera intrinsics** (fx, fy, ppx, ppy) langsung dari stream profile
- Menyusun `camera_matrix` 3×3:
  ```
  [[fx, 0,  cx],
   [0,  fy, cy],
   [0,  0,  1 ]]
  ```
- Batas maksimum frame: `MAX_FRAMES = 2000`
- Menangani akhir file dengan `RuntimeError` (perilaku normal pyrealsense2)

#### `services/video_generator.py` — `VideoFrameGenerator`

Membaca file `.mp4` menggunakan OpenCV.

- Konversi BGR → RGB
- `depth_map` diisi `np.zeros(...)` karena tidak ada data depth di MP4
- `camera_matrix` diisi `np.eye(3)` (identity/dummy)
- Timestamp dari `cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0` (dalam detik)

---

### 5.4 Core — Pipeline

#### `core/pipeline.py` — `PerceptionPipeline`

Kelas inti yang mengorkestrasi seluruh pemrosesan per frame. Diinisialisasi satu kali per run dan memegang referensi ke semua model serta repository.

**Inisialisasi:**
```python
self.detector = YOLODetector(model_path)   # Model YOLO
self.depth    = StereoDepth()              # Depth estimator
self.flow     = GlobalOpticalFlow()        # Optical flow
self.risk_engine = RiskEngine()            # Risk calculator
```

**Per Frame (`process_frame`):**
1. Simpan metadata frame ke DB via `FrameRepository`
2. Jalankan YOLO detection → untuk tiap objek: hitung jarak + risk → simpan ke DB
3. Hitung optical flow → simpan ke DB
4. Hitung scene risk (jumlah HIGH object) → simpan ke DB
5. Gambar anotasi pada frame → simpan PNG via `FrameSaver`

**Return value tiap frame:**
```python
{
    "frame_id": int,
    "detections": [{"frame_id", "class_id", "confidence", "bbox",
                    "distance_m", "risk"}, ...],
    "flow": {"mean_magnitude", "median_magnitude", "std_magnitude",
             "mean_dx", "mean_dy"} | None,
    "scene_risk": int,  # jumlah HIGH risk objects
    "alert": bool       # True jika scene_risk > 0
}
```

**Anotasi Visual (`_draw_annotations`):**
- Bounding box berwarna: **Merah** = HIGH, **Kuning** = MEDIUM, **Hijau** = LOW
- Label: `{class_id} | {distance}m | {risk_level}`
- Info optical flow di pojok kiri atas: `Flow: {mean_magnitude:.2f}`
- Info scene risk: `Scene Risk: {scene_risk_score}`

---

### 5.5 Core — Detection

#### `core/detection/base_detector.py` — `BaseDetector`

Abstract base class dengan satu method yang wajib diimplementasi:
```python
def detect(self, image): raise NotImplementedError
```

#### `core/detection/yolo_detector.py` — `YOLODetector`

Wrapper untuk model YOLO dari library `ultralytics`.

- **Confidence threshold:** 0.25 (default)
- **Input:** RGB image (numpy array)
- **Output:** List of dicts:
  ```python
  {
      "bbox": [x1, y1, x2, y2],   # format xyxy
      "confidence": float,
      "class_id": int
  }
  ```

---

### 5.6 Core — Depth

#### `core/depth/stereo_depth.py` — `StereoDepth`

Mengekstrak estimasi jarak objek dari depth map stereo.

**Metode `compute_distance(depth_map, bbox)`:**
1. Potong ROI dari `depth_map` menggunakan koordinat bounding box
2. Filter: ambil hanya pixel yang **finite** dan **> 0** (membuang data tidak valid)
3. Butuh minimal **10 pixel valid**; jika kurang → return `None`
4. Return **median** dari pixel valid (dalam satuan meter untuk data RealSense, atau raw unit untuk format lain)

> **Catatan:** Untuk input `.mp4`, `depth_map` berisi semua nol sehingga semua jarak akan `None` dan risk akan `UNKNOWN`.

---

### 5.7 Core — Optical Flow

#### `core/optical_flow/global_flow.py` — `GlobalOpticalFlow`

Menghitung pergerakan global seluruh scene menggunakan **dense optical flow Farnebäck**.

**Parameter Farnebäck:**
| Parameter | Nilai |
|---|---|
| `pyr_scale` | 0.5 |
| `levels` | 3 |
| `winsize` | 15 |
| `iterations` | 3 |
| `poly_n` | 5 |
| `poly_sigma` | 1.2 |

**Output (per frame, kecuali frame pertama yang return `None`):**
```python
{
    "mean_magnitude":   float,  # rata-rata kecepatan piksel
    "median_magnitude": float,  # median kecepatan piksel
    "std_magnitude":    float,  # standar deviasi kecepatan
    "mean_dx":          float,  # rata-rata pergerakan horizontal
    "mean_dy":          float   # rata-rata pergerakan vertikal
}
```

State `prev_gray` direset setiap kali `pipeline.reset()` dipanggil (setiap run baru).

---

### 5.8 Core — Risk Engine

#### `core/calculation/risk_engine.py` — `RiskEngine`

**`estimate_risk(distance_m)` — Risk per objek:**

| Jarak | Level Risiko |
|---|---|
| `None` | `UNKNOWN` |
| `< 5 meter` | `HIGH` |
| `5 – 10 meter` | `MEDIUM` |
| `≥ 10 meter` | `LOW` |

**`compute_scene_risk(object_calcs)` — Risk level scene:**

Menghitung **jumlah objek berstatus HIGH** pada frame tersebut. Nilai ini menjadi `scene_risk_score` di tabel `scene_metrics`, dan jika score > 0 maka `alert_flag = 1`.

---

### 5.9 Core — Video Output

#### `core/video/frame_saver.py` — `FrameSaver`

Menyimpan frame yang sudah dianotasi sebagai file PNG.

- Direktori: `assets/temp_frames/{run_id}/`
- Nama file: `frame_{frame_id:06d}.png` (zero-padded 6 digit)
- Direktori otomatis dihapus dan dibuat ulang di awal setiap run (overwrite)

#### `core/video/video_builder.py` — `VideoBuilder`

Merakit file PNG yang tersimpan menjadi satu file video MP4.

- Mencari semua `frame_*.png` di direktori input, diurutkan
- Codec: diambil dari konfigurasi (`VIDEO_CONFIG`)
- FPS: diambil dari source jika ada, fallback ke konfigurasi
- Dimensi video: diambil dari frame pertama secara otomatis

Setelah video silent selesai dibuat, `VideoBuilder.attach_audio()` melakukan mux audio alert WAV ke MP4 menggunakan FFmpeg.

#### `core/video/audio_alert.py` — `generate_alert_wav()`

Menghasilkan file audio WAV dengan sinyal peringatan.

| Parameter | Nilai |
|---|---|
| Sample rate | 44100 Hz |
| Frekuensi beep | 1200 Hz |
| Pulsa per detik | 3 |
| Duty cycle | 35% |
| Volume | 35% |

- Satu segmen audio dihasilkan per frame
- Jika `alert = False` → silence; jika `alert = True` → beep
- Durasi segmen = `1.0 / fps` detik

---

### 5.10 Database

#### `database/db.py`

| Fungsi | Keterangan |
|---|---|
| `get_connection()` | Buka koneksi SQLite ke `perception.db` |
| `init_database()` | Buat tabel jika belum ada (mengeksekusi `schema.sql`) |
| `reset_database()` | Hapus isi 4 tabel (`DELETE FROM`) tanpa menghapus struktur tabel |

Path database: `assistance_car/perception.db` (relatif terhadap root modul).

#### `database/repository.py`

Lima kelas repository mengikuti pola **Repository Pattern** untuk memisahkan logika akses data:

| Repository | Operasi | Tabel |
|---|---|---|
| `FrameRepository` | `insert(frame_data)` | `frames` |
| `DetectionRepository` | `insert(det)` | `detections` |
| `OpticalFlowRepository` | `insert(frame_id, flow_stats)` | `optical_flow` |
| `SceneRepository` | `insert(frame_id, scene_risk, alert_flag)` | `scene_metrics` |
| `AnalyticsRepository` | `summary()` | Semua tabel (SELECT) |

`AnalyticsRepository.summary()` mengembalikan:
```python
{
    "total_frames":     int,
    "total_detections": int,
    "high_risk_objects": int,
    "avg_flow":         float | None
}
```

---

### 5.11 Utils

#### `utils/frame_models.py` — `FrameData`

Dataclass yang menjadi "kontrak data" antar komponen:

```python
@dataclass
class FrameData:
    frame_id:      int
    timestamp:     float
    rgb_image:     np.ndarray        # (H, W, 3) RGB
    depth_map:     np.ndarray        # (H, W) uint16 atau float
    camera_matrix: np.ndarray        # (3, 3)
    image_path:    Optional[str] = None
    depth_path:    Optional[str] = None
    detections:    List[Dict] = []
    flow_stats:    Optional[Dict] = None
    scene_risk:    Optional[float] = None
```

#### `utils/logger.py` — `get_logger(name)`

- Level logging: `INFO`
- Output: file `logs/system.log` **dan** console (stdout)
- Format: `{timestamp} | {level} | {name} | {message}`
- Logger di-cache per nama untuk menghindari duplikasi handler

---

## 6. Skema Database

Database: `perception.db` (SQLite)

### Tabel `frames`
Menyimpan metadata setiap frame yang diproses.

| Kolom | Tipe | Keterangan |
|---|---|---|
| `frame_id` | INTEGER PK | Indeks frame (mulai dari 0) |
| `timestamp` | REAL | Waktu frame (detik) |
| `image_path` | TEXT | Path ke image asli (null saat ini) |
| `depth_path` | TEXT | Path ke depth map asli (null saat ini) |
| `fx` | REAL | Focal length X (dari intrinsics kamera) |
| `fy` | REAL | Focal length Y |
| `cx` | REAL | Principal point X |
| `cy` | REAL | Principal point Y |

### Tabel `detections`
Menyimpan setiap objek yang terdeteksi beserta estimasi jarak dan level risikonya.

| Kolom | Tipe | Keterangan |
|---|---|---|
| `id` | INTEGER PK AUTOINCREMENT | ID unik deteksi |
| `frame_id` | INTEGER FK | Referensi ke tabel frames |
| `class_id` | INTEGER | ID kelas objek dari YOLO |
| `confidence` | REAL | Skor kepercayaan YOLO (0–1) |
| `bbox_x1` | REAL | Koordinat kiri bounding box |
| `bbox_y1` | REAL | Koordinat atas bounding box |
| `bbox_x2` | REAL | Koordinat kanan bounding box |
| `bbox_y2` | REAL | Koordinat bawah bounding box |
| `distance_m` | REAL | Jarak estimasi ke objek (meter) |
| `risk_level` | TEXT | `HIGH` / `MEDIUM` / `LOW` / `UNKNOWN` |

### Tabel `optical_flow`
Menyimpan statistik optical flow global per frame.

| Kolom | Tipe | Keterangan |
|---|---|---|
| `frame_id` | INTEGER PK FK | Referensi ke tabel frames |
| `mean_magnitude` | REAL | Rata-rata besaran pergerakan piksel |
| `median_magnitude` | REAL | Median besaran pergerakan piksel |
| `std_magnitude` | REAL | Standar deviasi besaran pergerakan |
| `mean_dx` | REAL | Rata-rata pergerakan horizontal |
| `mean_dy` | REAL | Rata-rata pergerakan vertikal |

### Tabel `scene_metrics`
Menyimpan penilaian risiko tingkat scene per frame.

| Kolom | Tipe | Keterangan |
|---|---|---|
| `frame_id` | INTEGER PK FK | Referensi ke tabel frames |
| `scene_risk_score` | REAL | Jumlah objek HIGH risk di frame ini |
| `alert_flag` | INTEGER | `1` jika scene_risk_score > 0, `0` jika tidak |

**Diagram Relasi:**
```
frames (frame_id PK)
  ├── detections.frame_id (FK)
  ├── optical_flow.frame_id (FK, PK)
  └── scene_metrics.frame_id (FK, PK)
```

---

## 7. Output Artifacts

Setiap kali memproses file, sistem menghasilkan beberapa output:

| Artifact | Path | Keterangan |
|---|---|---|
| **Video anotasi** | `assets/output/{run_id}.mp4` | Video dengan bounding box, jarak, risk level |
| **Audio alert** | `assets/output/{run_id}_alert.wav` | File WAV dengan beep saat HIGH risk |
| **Frame sementara** | `assets/temp_frames/{run_id}/frame_XXXXXX.png` | Dibuat ulang tiap run |
| **Database** | `perception.db` | Data lengkap semua analisis (di-reset tiap run) |
| **Log** | `logs/system.log` | Log detail setiap langkah pemrosesan |

> `run_id` diambil dari nama file input tanpa ekstensi (misalnya, input `recording.bag` → `run_id = recording`).

---

## 8. Cara Menjalankan

### Prasyarat

```bash
pip install ultralytics opencv-python numpy fastapi "uvicorn[standard]" python-multipart pyrealsense2 imageio-ffmpeg
```

Pastikan `ffmpeg` tersedia di PATH (atau set command FFmpeg lewat `config/settings.py`) untuk menggabungkan audio alert ke video output. Jika FFmpeg tidak tersedia di PATH, sistem akan mencoba fallback ke binary dari `imageio-ffmpeg`.

Pastikan file model YOLO tersedia di `assets/models/yolo.pt`.

### Opsi 1: FastAPI Web UI

```bash
cd assistance_car
uvicorn app:app --reload
```

Buka browser di `http://127.0.0.1:8000`, lalu:
1. Pilih **Source Type** (`bag` atau `mp4`)
2. Upload file `.mp4` atau masukkan path absolut file `.bag`
3. Klik **Process**
4. Tunggu proses selesai
5. Lihat **Perception Summary**, preview video output, dan audio alert

### Opsi 2: CLI (Backend Only)

```bash
cd assistance_car

# Proses file .bag
python run_backend.py --source "C:/path/to/recording.bag" --type bag

# Proses file .mp4
python run_backend.py --source "C:/path/to/video.mp4" --type mp4
```

---

## 9. Dependensi

| Library | Fungsi |
|---|---|
| `ultralytics` | YOLO v11 object detection |
| `pyrealsense2` | Membaca file `.bag` Intel RealSense |
| `opencv-python` | Pemrosesan gambar, optical flow, video I/O |
| `numpy` | Komputasi numerik array |
| `fastapi` | Web framework untuk API + HTML UI sederhana |
| `uvicorn` | ASGI server untuk menjalankan FastAPI |
| `python-multipart` | Parsing upload file dari form HTML |
| `ffmpeg` | Mux audio alert WAV ke video MP4 |
| `imageio-ffmpeg` | Fallback binary FFmpeg jika FFmpeg sistem tidak tersedia |
| `sqlite3` | Database (stdlib Python) |
| `wave` | Generate file audio WAV (stdlib Python) |
| `dataclasses` | FrameData model (stdlib Python) |
| `logging` | Sistem logging (stdlib Python) |

---

## 10. Catatan Pengembangan Lanjutan

Berdasarkan README awal dan kode yang sudah ada, berikut komponen yang sudah tersedia strukturnya namun belum diimplementasi sepenuhnya:

| Komponen | Status | Keterangan |
|---|---|---|
| `core/frame_processor.py` | Kosong | Placeholder untuk future frame preprocessing |
| `core/lane/` | Kosong | Dirancang untuk lane detection |
| `core/detection/rf_detr_detector.py` | Belum ada | Alternatif detector (RF-DETR) |
| `core/depth/depth_anything.py` | Belum ada | Depth estimation monocular (tanpa stereo) |
| `core/optical_flow/object_flow.py` | Belum ada | Per-object optical flow (bukan global) |
| `core/calculation/fusion_engine.py` | Belum ada | Fusion dari berbagai sinyal risk |
| `core/calculation/metrics.py` | Belum ada | Metrics evaluation |
| Output video playback di UI | Aktif | Preview video + audio tersedia di halaman FastAPI |
| Halaman dashboard multi-page | Belum ada | `ui/pages/` dari arsitektur target |
