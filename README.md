# 🚗 Assistance Car Perception System

Sistem persepsi kendaraan berbasis computer vision yang memproses video dari kamera stereo (Intel RealSense `.bag`) atau file video biasa (`.mp4`) untuk melakukan **object detection**, **depth estimation**, **global + object optical flow**, dan **risk assessment** secara real-time per frame. Hasil analisis disimpan ke database SQLite dan ditampilkan melalui antarmuka FastAPI Web UI sederhana (HTML).

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
| **Optical Flow** | Menganalisis pergerakan global scene dan pergerakan tiap objek (ROI per-bounding-box) dari dense flow Farnebäck |
| **Lane + Drivable Segmentation** | Segmentasi lajur dan area drivable menggunakan TwinLiteNetPlus |
| **Risk Assessment** | Menilai level risiko setiap objek (HIGH / MEDIUM / LOW / UNKNOWN) dan risiko keseluruhan scene dengan mempertimbangkan status gerak objek |
| **Persistensi Data** | Menyimpan semua hasil analisis ke database SQLite (`perception.db`) |
| **Output Video** | Menghasilkan video anotasi dengan bounding box, jarak, level risiko, status gerak objek (MOV/STA), panah gerak, info scene flow, serta overlay inference time + FPS |
| **Audio Alert** | Menghasilkan file WAV dengan bunyi beep saat terdeteksi objek berisiko tinggi |
| **Performance Evaluation** | Mengukur inference time per fitur, total pipeline latency, FPS, latency percentile (P50/P95), dan rata-rata deteksi per frame |
| **Dashboard** | Menampilkan ringkasan analitik (risk + performance) melalui antarmuka FastAPI Web UI |

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
│         │                 │          ┌────────▼──────────┐  │
│         │                 │          │ ObjectOpticalFlow │  │
│         │                 │          │ (per-bbox ROI     │  │
│         │                 │          │  motion stats)    │  │
│         │                 │          └────────┬──────────┘  │
│         └─────────────────▼───────────────────▼────────────┐│
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
│   frames | detections | optical_flow | scene_metrics | lane_metrics | performance_metrics │
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
│   ├── lane/                       # Lane segmentation (TwinLiteNetPlus)
│   │   ├── lane_detector.py        # Wrapper inferensi lane/drivable
│   │   └── twinlitenet_model.py    # Arsitektur model TwinLiteNetPlus
│   │
│   ├── optical_flow/
│   │   ├── global_flow.py          # Dense flow global (scene level)
│   │   └── object_flow.py          # Statistik flow per objek (ROI bbox)
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
│   ├── schema.sql                  # DDL: tabel analytics + performance
│   └── repository.py               # Repository pattern: 6 kelas repository
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
│  Langkah 3: Optical Flow                    │
│    → GlobalOpticalFlow.compute(rgb_image)   │
│      Metode: Farnebäck dense flow          │
│      Output: (flow_stats, flow_field)       │
│    → OpticalFlowRepository.insert()        │
│      → INSERT INTO optical_flow            │
│                                             │
│  Langkah 4: Object Optical Flow             │
│    → ObjectOpticalFlow.compute_object_flows(│
│        flow_field, detections)              │
│      Output per objek:                      │
│      {object_magnitude, object_dx,          │
│       object_dy, is_moving}                 │
│                                             │
│  Langkah 5: Depth + Risk per Object         │
│    → StereoDepth.compute_distance(...)       │
│    → RiskEngine.assess_object_risk(          │
│        distance_m, object_flow, lane_result, │
│        bbox, class_id)                      │
│      Output: risk + risk_score +             │
│      lane_overlap + occupancy contribution   │
│    → DetectionRepository.insert()           │
│      → INSERT INTO detections               │
│        (+ risk_score, lane_overlap, object_flow) │
│                                             │
│  Langkah 6: Scene Fusion Metrics            │
│    → RiskEngine.compute_scene_metrics()     │
│      Output: scene_risk_score,              │
│      path_occupancy_risk, dynamic_hazard,   │
│      drivable_capacity, trip_safety, alert  │
│    → SceneRepository.insert(frame_id, metrics) │
│      → INSERT INTO scene_metrics            │
│                                             │
│  Langkah 7: Simpan Frame Anotasi            │
│    → _draw_annotations() → BGR image        │
│      Anotasi: bounding box berwarna,        │
│               label class + jarak + risk,   │
│               MOV/STA + panah gerak objek,  │
│               info scene flow + scene risk, │
│               inference ms + FPS            │
│    → FrameSaver.save() → PNG               │
│      Path: assets/temp_frames/{run_id}/     │
│                                             │
│  Langkah 8: Simpan Performance Metrics       │
│    → INSERT INTO performance_metrics         │
│      (per-model ms, total ms, fps, p95-ready)|
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
3. Ringkasan metrik operasional + insight (`Path Occupancy Risk`, `Dynamic Hazard Index`, `Drivable Capacity Score`, `Trip Safety Score`)
4. Preview video output dan audio alert

Alur di `app.py`:
1. `init_database()` dipanggil saat startup FastAPI
2. Saat endpoint `/process` dipanggil → file MP4 disimpan sementara (temporary file) atau path BAG langsung digunakan
3. `VideoService(YOLO_MODEL_PATH).process(source_path, source_type)` dijalankan
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
YOLO_MODEL_PATH = BASE_DIR / "assets" / "models" / "yolo.pt"
```

Konfigurasi penting sekarang mencakup:
- `YOLO_MODEL_PATH` dan `LANE_MODEL_PATH`
- `RISK_CONFIG` untuk threshold jarak
- `RISK_FUSION_CONFIG` untuk bobot fusion object/scene insights
- `FLOW_CONFIG` dan `OBJECT_FLOW_CONFIG` untuk optical flow global + object-level

---

### 5.3 Services

#### `services/video_service.py` — `VideoService`

Orkestrator level "per run" — satu call ke `process()` menangani satu file input secara penuh.

Langkah-langkah `process()`:
1. `reset_database()` — Hapus data lama dari 5 tabel
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
- Batas maksimum frame: dari `PROCESSING_CONFIG["max_bag_frames"]` (default saat ini: `20000`)
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
self.flow     = GlobalOpticalFlow()        # Global optical flow (scene level)
self.object_flow = ObjectOpticalFlow()     # Object flow (per-bbox ROI)
self.risk_engine = RiskEngine()            # Risk calculator
```

**Per Frame (`process_frame`):**
1. Simpan metadata frame ke DB via `FrameRepository`
2. Jalankan YOLO detection
3. Hitung global optical flow (`flow_stats` + `flow_field`) → simpan `flow_stats` ke DB
4. Hitung object flow per deteksi dengan slicing ROI pada `flow_field`
5. Jalankan lane segmentation (bila aktif) → simpan ke DB
6. Untuk tiap objek: hitung jarak, lane overlap, motion score, lalu risk fusion per objek → simpan ke DB
7. Hitung scene fusion metrics (`scene_risk_score`, `path_occupancy_risk`, `dynamic_hazard_index`, `drivable_capacity_score`, `trip_safety_score`) → simpan ke DB
8. Gambar anotasi pada frame → simpan PNG via `FrameSaver`

**Return value tiap frame:**
```python
{
    "frame_id": int,
    "detections": [{"frame_id", "class_id", "confidence", "bbox",
      "distance_m", "risk", "risk_score", "lane_overlap_ratio", "object_flow"}, ...],
    "flow": {"mean_magnitude", "median_magnitude", "std_magnitude",
             "mean_dx", "mean_dy"} | None,
  "lane": {"lane_pixel_ratio", "drivable_pixel_ratio", ...} | None,
  "scene_metrics": {
    "scene_risk_score", "path_occupancy_risk", "dynamic_hazard_index",
    "drivable_capacity_score", "trip_safety_score", "alert_flag"
  },
    "scene_risk": int,  # jumlah HIGH risk objects
    "alert": bool       # True jika scene_risk > 0
}
```

**Anotasi Visual (`_draw_annotations`):**
- Bounding box berwarna berdasarkan risk + status gerak (moving objek ditampilkan dengan tone berbeda)
- Label: `{class_id} | {distance}m | {risk_level} | MOV/STA` (MOV memuat magnitude px/frame)
- Panah gerak ditampilkan untuk objek moving
- Info optical flow di pojok kiri atas: `Scene Flow: {mean_magnitude:.2f} px/f`
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
(flow_stats, flow_field)

flow_stats = {
  "mean_magnitude":   float,
  "median_magnitude": float,
  "std_magnitude":    float,
  "mean_dx":          float,
  "mean_dy":          float
}

flow_field.shape == (H, W, 2)  # [dx, dy] per pixel
```

State `prev_gray` direset setiap kali `pipeline.reset()` dipanggil (setiap run baru).

#### `core/optical_flow/object_flow.py` — `ObjectOpticalFlow`

Menghitung pergerakan per objek dari `flow_field` global, dengan cara mengambil ROI sesuai bounding box setiap deteksi.

Karakteristik implementasi:
- **Stateless**: tidak menyimpan state antar frame
- Input selalu `flow_field` terbaru dari pipeline
- Koordinat bbox di-clamp ke batas frame untuk mencegah out-of-bound
- Jika ROI tidak valid, hasil object flow untuk objek tersebut adalah `None`

**Output per objek:**
```python
{
  "object_magnitude": float,  # mean magnitude di ROI bbox (px/frame)
  "object_dx":        float,
  "object_dy":        float,
  "is_moving":        bool    # object_magnitude > moving_threshold
}
```

Threshold gerak default di konfigurasi:
- `OBJECT_FLOW_CONFIG["moving_threshold"] = 1.5` px/frame

---

### 5.8 Core — Risk Engine

#### `core/calculation/risk_engine.py` — `RiskEngine`

Risk engine kini memakai **fusion scoring**, bukan sekadar downgrade rule.

**`assess_object_risk(distance_m, object_flow, lane_result, bbox, class_id)` — Risk per objek:**

Komponen per objek yang digabung:
- `proximity_score` (dari distance)
- `motion_score` (dari object flow magnitude + moving state)
- `lane_overlap_ratio` (overlap bbox terhadap drivable area)
- `class_weight` (bobot kelas objek)

Output object-context:
```python
{
  "risk": "HIGH|MEDIUM|LOW|UNKNOWN",
  "risk_score": float,
  "proximity_score": float,
  "motion_score": float,
  "lane_overlap_ratio": float,
  "path_occupancy_risk": float,
  "class_weight": float
}
```

`estimate_risk(...)` tetap tersedia sebagai wrapper yang mengembalikan label final risk.

**Threshold label object risk:**

| Kondisi Skor | Level Risiko |
|---|---|
| `risk_score >= object_high_threshold` | `HIGH` |
| `risk_score >= object_medium_threshold` | `MEDIUM` |
| selain itu | `LOW` / `UNKNOWN` (jika konteks tidak cukup) |

**`compute_scene_metrics(object_calcs, flow_stats, lane_result)` — Insight level scene:**

Metrik scene yang dihasilkan:
- `scene_risk_score`: jumlah objek berlabel `HIGH`
- `path_occupancy_risk`: akumulasi risiko objek pada area drivable
- `dynamic_hazard_index`: indeks hazard dinamis gabungan object risk + occupancy + flow
- `drivable_capacity_score`: kapasitas area drivable tersisa setelah dikurangi occupancy risk
- `trip_safety_score`: skor keselamatan agregat frame-level
- `alert_flag`: status alert gabungan rule scene

`compute_scene_risk(object_calcs)` tetap ada sebagai utilitas untuk menghitung jumlah objek HIGH risk.

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
| `reset_database()` | Hapus isi 5 tabel (`DELETE FROM`) tanpa menghapus struktur tabel |

Path database: `assistance_car/perception.db` (relatif terhadap root modul).

#### `database/repository.py`

Enam kelas repository mengikuti pola **Repository Pattern** untuk memisahkan logika akses data:

| Repository | Operasi | Tabel |
|---|---|---|
| `FrameRepository` | `insert(frame_data)` | `frames` |
| `DetectionRepository` | `insert(det)` | `detections` |
| `OpticalFlowRepository` | `insert(frame_id, flow_stats)` | `optical_flow` |
| `SceneRepository` | `insert(frame_id, scene_metrics)` | `scene_metrics` |
| `LaneRepository` | `insert(frame_id, lane_stats)` | `lane_metrics` |
| `AnalyticsRepository` | `summary()` | Semua tabel (SELECT) |

`AnalyticsRepository.summary()` mengembalikan:
```python
{
    "total_frames":     int,
    "total_detections": int,
    "high_risk_objects": int,
  "avg_object_risk_score": float | None,
  "avg_lane_overlap_ratio": float | None,
    "avg_flow":         float | None,
  "avg_scene_risk_score": float | None,
  "avg_path_occupancy_risk": float | None,
  "avg_dynamic_hazard_index": float | None,
  "avg_drivable_capacity_score": float | None,
  "avg_trip_safety_score": float | None,
    "avg_lane_ratio":   float | None,
    "avg_drivable_ratio": float | None
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
Menyimpan setiap objek yang terdeteksi beserta estimasi jarak, level risiko, dan metrik object flow.

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
| `risk_score` | REAL | Skor risiko objek hasil fusion |
| `lane_overlap_ratio` | REAL | Rasio overlap bbox dengan area drivable |
| `object_flow_magnitude` | REAL | Mean magnitude flow pada ROI objek (px/frame) |
| `object_flow_dx` | REAL | Rata-rata komponen horizontal flow ROI objek |
| `object_flow_dy` | REAL | Rata-rata komponen vertikal flow ROI objek |
| `is_moving` | INTEGER | `1` moving, `0` stationary, `NULL` jika flow belum tersedia |

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
| `path_occupancy_risk` | REAL | Risiko okupansi path berdasarkan objek pada drivable area |
| `dynamic_hazard_index` | REAL | Indeks hazard dinamis gabungan objek + flow + occupancy |
| `drivable_capacity_score` | REAL | Kapasitas area drivable tersisa |
| `trip_safety_score` | REAL | Skor keselamatan agregat frame-level |
| `alert_flag` | INTEGER | `1` jika alert rule terpenuhi, `0` jika tidak |

### Tabel `lane_metrics`
Menyimpan statistik segmentasi lane/drivable per frame.

| Kolom | Tipe | Keterangan |
|---|---|---|
| `frame_id` | INTEGER PK FK | Referensi ke tabel frames |
| `lane_pixel_ratio` | REAL | Persentase piksel lane terhadap frame |
| `drivable_pixel_ratio` | REAL | Persentase area drivable terhadap frame |

### Tabel `performance_metrics`
Menyimpan metrik performa inferensi per frame untuk evaluasi sistem.

| Kolom | Tipe | Keterangan |
|---|---|---|
| `frame_id` | INTEGER PK FK | Referensi ke tabel frames |
| `yolo_ms` | REAL | Waktu inferensi deteksi YOLO (ms) |
| `global_flow_ms` | REAL | Waktu komputasi global optical flow (ms) |
| `object_flow_ms` | REAL | Waktu komputasi object optical flow (ms) |
| `lane_ms` | REAL | Waktu inferensi lane/drivable segmentation (ms) |
| `risk_ms` | REAL | Waktu perhitungan depth + risk per object (ms) |
| `scene_ms` | REAL | Waktu perhitungan scene fusion metrics (ms) |
| `annotation_ms` | REAL | Waktu render anotasi frame (ms) |
| `pipeline_total_ms` | REAL | Total waktu proses pipeline per frame (ms) |
| `pipeline_fps` | REAL | FPS estimasi per frame ($FPS = 1000 / ms$) |
| `detection_count` | INTEGER | Jumlah objek terdeteksi pada frame |

**Diagram Relasi:**
```
frames (frame_id PK)
  ├── detections.frame_id (FK)
  ├── optical_flow.frame_id (FK, PK)
  ├── scene_metrics.frame_id (FK, PK)
  ├── lane_metrics.frame_id (FK, PK)
  └── performance_metrics.frame_id (FK, PK)
```

### Metrik yang Ditampilkan

UI FastAPI (card Performance Evaluation):
- Inference time per fitur/model: YOLO, global flow, object flow, lane, risk, scene, annotation
- Inference keseluruhan: average total latency, min/max latency
- Distribusi latency: P50 dan P95 total latency
- Throughput: average FPS
- Kompleksitas scene: average detection per frame

Overlay video/realtime:
- Per-frame total inference time (ms)
- Per-frame FPS

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

Berikut status komponen lanjutan saat ini:

| Komponen | Status | Keterangan |
|---|---|---|
| `core/frame_processor.py` | Kosong | Placeholder untuk future frame preprocessing |
| `core/lane/` | Aktif | TwinLiteNetPlus lane + drivable segmentation sudah terintegrasi |
| `core/detection/rf_detr_detector.py` | Belum ada | Alternatif detector (RF-DETR) |
| `core/depth/depth_anything.py` | Belum ada | Depth estimation monocular (tanpa stereo) |
| `core/optical_flow/object_flow.py` | Aktif | Object-level flow stateless berbasis ROI bbox |
| `core/calculation/fusion_engine.py` | Belum ada | Fusion dari berbagai sinyal risk |
| `core/calculation/metrics.py` | Belum ada | Metrics evaluation |
| Output video playback di UI | Aktif | Preview video + audio tersedia di halaman FastAPI |
| Halaman dashboard multi-page | Belum ada | `ui/pages/` dari arsitektur target |
