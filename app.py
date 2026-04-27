from __future__ import annotations

from html import escape
from pathlib import Path
import shutil
import tempfile
import time

from fastapi import FastAPI, File, Form, UploadFile, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from config.settings import YOLO_MODEL_PATH, OUTPUT_DIR, UI_CONFIG
from database.db import init_database
from database.repository import AnalyticsRepository
from services.realtime_stream_service import RealSenseRealtimeService
from services.video_service import VideoService
from utils.logger import get_logger


app = FastAPI(title=UI_CONFIG["title"])
app.mount("/media/output", StaticFiles(directory=str(OUTPUT_DIR)), name="media-output")
realtime_service = RealSenseRealtimeService(str(YOLO_MODEL_PATH))
app_logger = get_logger("WebApp")


# ── Live camera stream (MJPEG) ──────────────────────────────────────────────

def _generate_camera_frames():
    """Yield MJPEG frames from the default webcam."""
    try:
        import cv2  # type: ignore
    except ImportError:
        raise RuntimeError("opencv-python is required for live stream. Run: pip install opencv-python")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Tidak dapat membuka kamera. Pastikan kamera terhubung.")
    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buffer.tobytes()
                + b"\r\n"
            )
    finally:
        cap.release()


@app.get("/camera/stream")
def camera_stream():
    """MJPEG endpoint consumed by the <img> tag in the UI."""
    return StreamingResponse(
        _generate_camera_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/realtime/stream")
def realtime_stream():
    """MJPEG endpoint for processed RealSense realtime stream."""
    return StreamingResponse(
        realtime_service.stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ── Video file streaming with Range-Request support (required by Chrome) ────

@app.get("/video/{filename}")
async def stream_video(filename: str, request: Request):
    """
    Serve a video file with proper HTTP 206 Partial Content support so that
    Chrome (and all modern browsers) can seek/scrub the video timeline.
    StaticFiles does NOT set Accept-Ranges correctly for all Chrome versions,
    so we handle it explicitly here.
    """
    video_path = OUTPUT_DIR / filename
    if not video_path.exists() or not video_path.is_file():
        from fastapi.responses import Response
        return Response(status_code=404)

    file_size = video_path.stat().st_size
    range_header = request.headers.get("range")

    def iter_file(path: Path, start: int, end: int, chunk: int = 1 << 20):
        with open(path, "rb") as fh:
            fh.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                data = fh.read(min(chunk, remaining))
                if not data:
                    break
                remaining -= len(data)
                yield data

    if range_header:
        raw_start, _, raw_end = range_header.replace("bytes=", "").partition("-")
        start = int(raw_start)
        end = int(raw_end) if raw_end else file_size - 1
    else:
        start, end = 0, file_size - 1

    end = min(end, file_size - 1)
    content_length = end - start + 1

    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(content_length),
        "Content-Disposition": f'inline; filename="{filename}"',
    }

    return StreamingResponse(
        iter_file(video_path, start, end),
        status_code=206,
        headers=headers,
        media_type="video/mp4",
    )


# ── HTML renderer ────────────────────────────────────────────────────────────

def _render_page(
    *,
    message: str = "",
    output_video_name: str = "",
    elapsed_seconds: float | None = None,
):
    analytics = AnalyticsRepository()
    summary = analytics.summary()

    total_frames = summary.get("total_frames", 0)
    total_detections = summary.get("total_detections", 0)
    high_risk_objects = summary.get("high_risk_objects", 0)
    avg_object_risk_score = summary.get("avg_object_risk_score", 0) or 0
    avg_lane_overlap_ratio = summary.get("avg_lane_overlap_ratio", 0) or 0
    avg_flow = summary.get("avg_flow", 0) or 0
    avg_scene_risk_score = summary.get("avg_scene_risk_score", 0) or 0
    avg_path_occupancy_risk = summary.get("avg_path_occupancy_risk", 0) or 0
    avg_dynamic_hazard_index = summary.get("avg_dynamic_hazard_index", 0) or 0
    avg_drivable_capacity_score = summary.get("avg_drivable_capacity_score", 0) or 0
    avg_trip_safety_score = summary.get("avg_trip_safety_score", 0) or 0
    avg_lane_ratio = summary.get("avg_lane_ratio", 0) or 0
    avg_drivable_ratio = summary.get("avg_drivable_ratio", 0) or 0
    avg_yolo_ms = summary.get("avg_yolo_ms", 0) or 0
    avg_global_flow_ms = summary.get("avg_global_flow_ms", 0) or 0
    avg_object_flow_ms = summary.get("avg_object_flow_ms", 0) or 0
    avg_lane_ms = summary.get("avg_lane_ms", 0) or 0
    avg_risk_ms = summary.get("avg_risk_ms", 0) or 0
    avg_scene_ms = summary.get("avg_scene_ms", 0) or 0
    avg_annotation_ms = summary.get("avg_annotation_ms", 0) or 0
    avg_pipeline_total_ms = summary.get("avg_pipeline_total_ms", 0) or 0
    p50_pipeline_total_ms = summary.get("p50_pipeline_total_ms", 0) or 0
    p95_pipeline_total_ms = summary.get("p95_pipeline_total_ms", 0) or 0
    min_pipeline_total_ms = summary.get("min_pipeline_total_ms", 0) or 0
    max_pipeline_total_ms = summary.get("max_pipeline_total_ms", 0) or 0
    avg_pipeline_fps = summary.get("avg_pipeline_fps", 0) or 0
    avg_detection_per_frame = summary.get("avg_detection_per_frame", 0) or 0

    escaped_message = escape(message)

    elapsed_text = ""
    if elapsed_seconds is not None:
        elapsed_text = f"<p><strong>Processing time:</strong> {elapsed_seconds:.2f} seconds</p>"

    # ── video panel ──────────────────────────────────────────────────────────
    # Uses the dedicated /video/<filename> endpoint (Range-Request aware) so
    # Chrome can play, seek, and scrub the output clip without issues.
    video_html = ""
    if output_video_name:
        safe_name = escape(output_video_name)
        video_src = f"/video/{safe_name}"
        download_src = f"/media/output/{safe_name}"
        video_html = f"""
        <h3>Output Video (dengan Audio Alert)</h3>
        <video
          id="output-video"
          controls
          width="840"
          preload="metadata"
          style="background:#000; display:block; max-width:100%; border-radius:8px;"
        >
          <source src="{video_src}" type="video/mp4">
          Browser Anda tidak support HTML5 video.
        </video>
        <p><a href="{download_src}" download>⬇ Download video</a></p>
        """

    return f"""
<!doctype html>
<html lang='en'>
<head>
  <meta charset='UTF-8'>
  <meta name='viewport' content='width=device-width, initial-scale=1.0'>
  <title>{escape(UI_CONFIG['title'])}</title>
  <style>
    :root {{
      --bg: #f2f6f5;
      --card: #ffffff;
      --ink: #1e2a27;
      --muted: #5f6f6a;
      --accent: #0f766e;
      --line: #d8e3e0;
    }}
    body {{
      margin: 0;
      font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
      background: linear-gradient(180deg, #e9f5f2 0%, var(--bg) 35%, #f9fbfa 100%);
      color: var(--ink);
    }}
    .container {{
      max-width: 1080px;
      margin: 0 auto;
      padding: 28px 18px 48px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 20px;
      box-shadow: 0 10px 26px rgba(22, 40, 36, 0.06);
      margin-bottom: 18px;
    }}
    h1, h2, h3 {{ margin: 8px 0 14px; }}
    .muted {{ color: var(--muted); margin-top: 0; }}
    .row {{ display: flex; flex-wrap: wrap; gap: 12px; }}
    .metric {{
      flex: 1 1 180px;
      background: #f5faf9;
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 12px;
    }}
    label {{ display: block; margin: 10px 0 6px; font-weight: 600; }}
    input[type='text'], select, input[type='file'] {{
      width: 100%;
      box-sizing: border-box;
      border: 1px solid #c7d8d4;
      border-radius: 8px;
      padding: 10px;
      background: #fff;
    }}
    button {{
      margin-top: 14px;
      border: none;
      background: var(--accent);
      color: #fff;
      padding: 10px 16px;
      border-radius: 8px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.3s ease;
    }}
    button:hover:not(:disabled) {{ opacity: 0.9; }}
    button:disabled {{
      background: #999;
      cursor: not-allowed;
      opacity: 0.6;
    }}
    .spinner {{
      display: inline-block;
      width: 16px;
      height: 16px;
      border: 3px solid rgba(255,255,255,.3);
      border-radius: 50%;
      border-top-color: #fff;
      animation: spin 0.6s linear infinite;
      margin-right: 8px;
      vertical-align: middle;
    }}
    @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
    .loading-indicator {{
      display: none;
      padding: 12px;
      background: #e3f2fd;
      border: 1px solid #90caf9;
      border-radius: 8px;
      color: #1565c0;
      margin-top: 10px;
      font-weight: 500;
    }}
    .loading-indicator.active {{ display: block; }}
    .notice {{
      padding: 10px 12px;
      border-radius: 8px;
      background: #ecfdf5;
      border: 1px solid #b7ead2;
      color: #135c4a;
    }}
    .error {{
      background: #fef2f2;
      border-color: #f8c5c5;
      color: #9a2f2f;
    }}
    video, img.live-feed {{ max-width: 100%; border-radius: 8px; }}

    /* ── View-mode tabs ── */
    .view-tabs {{
      display: flex;
      gap: 8px;
      margin-bottom: 16px;
    }}
    .view-tab {{
      padding: 8px 20px;
      border-radius: 8px;
      border: 1px solid var(--line);
      background: #f5faf9;
      color: var(--ink);
      font-weight: 600;
      cursor: pointer;
      transition: background 0.2s, color 0.2s;
    }}
    .view-tab.active {{
      background: var(--accent);
      color: #fff;
      border-color: var(--accent);
    }}
    .view-panel {{ display: none; }}
    .view-panel.active {{ display: block; }}

    /* camera status badge */
    .cam-badge {{
      display: inline-block;
      padding: 3px 10px;
      border-radius: 20px;
      font-size: 0.78rem;
      font-weight: 700;
      background: #fee2e2;
      color: #b91c1c;
      margin-left: 10px;
      vertical-align: middle;
    }}
    .cam-badge.live {{
      background: #dcfce7;
      color: #15803d;
    }}
  </style>
</head>
<body>
  <div class='container'>

    <div class='card'>
      <h1>{escape(UI_CONFIG['title'])}</h1>
      <p class='muted'>{escape(UI_CONFIG['max_upload_size_note'])}</p>
    </div>

    <div class='card'>
      <h2>Process Source</h2>
      <form id='process-form' method='post' action='/process' enctype='multipart/form-data'>
        <label for='source_type'>Source Type</label>
        <select id='source_type' name='source_type'>
          <!-- <option value='mp4'>mp4</option> -->
          <option value='bag'>bag</option>
        </select>

        <label for='bag_file'>Upload BAG File (opsional, untuk bag source)</label>
        <input id='bag_file' name='bag_file' type='file' accept='.bag'>

        <!--
        <label for='file'>Upload MP4 File (for mp4 source)</label>
        <input id='file' name='file' type='file' accept='.mp4'>
        -->

        <label for='bag_path'>Local BAG Path (for bag source)</label>
        <input id='bag_path' name='bag_path' type='text' placeholder='C:/path/to/file.bag'>

        <button id='process-button' type='submit'>Process</button>
        <div id='loading-indicator' class='loading-indicator'>
          <span class='spinner'></span>
          <span>Processing video... Silakan tunggu...</span>
        </div>
      </form>
    </div>

    <div class='card'>
      <h2>Perception Summary</h2>
      <div class='row'>
        <div class='metric'><strong>Total Frames</strong><br>{total_frames}</div>
        <div class='metric'><strong>Total Detections</strong><br>{total_detections}</div>
        <div class='metric'><strong>High Risk Objects</strong><br>{high_risk_objects}</div>
        <div class='metric'><strong>Avg Object Risk</strong><br>{avg_object_risk_score:.2f}</div>
        <div class='metric'><strong>Avg Lane Overlap</strong><br>{avg_lane_overlap_ratio * 100:.2f}%</div>
        <div class='metric'><strong>Average Flow</strong><br>{avg_flow:.2f}</div>
        <div class='metric'><strong>Avg Scene Risk</strong><br>{avg_scene_risk_score:.2f}</div>
        <div class='metric'><strong>Path Occupancy Risk</strong><br>{avg_path_occupancy_risk:.2f}</div>
        <div class='metric'><strong>Dynamic Hazard Index</strong><br>{avg_dynamic_hazard_index:.2f}</div>
        <div class='metric'><strong>Drivable Capacity</strong><br>{avg_drivable_capacity_score:.2f}</div>
        <div class='metric'><strong>Trip Safety Score</strong><br>{avg_trip_safety_score:.2f}</div>
        <div class='metric'><strong>Avg Lane Pixels</strong><br>{avg_lane_ratio * 100:.2f}%</div>
        <div class='metric'><strong>Avg Drivable Area</strong><br>{avg_drivable_ratio * 100:.2f}%</div>
      </div>
    </div>

    <div class='card'>
      <h2>Performance Evaluation</h2>
      <div class='row'>
        <div class='metric'><strong>Avg Total Inference</strong><br>{avg_pipeline_total_ms:.2f} ms</div>
        <div class='metric'><strong>Avg FPS</strong><br>{avg_pipeline_fps:.2f}</div>
        <div class='metric'><strong>P50 Total Latency</strong><br>{p50_pipeline_total_ms:.2f} ms</div>
        <div class='metric'><strong>P95 Total Latency</strong><br>{p95_pipeline_total_ms:.2f} ms</div>
        <div class='metric'><strong>Min/Max Total Latency</strong><br>{min_pipeline_total_ms:.2f} / {max_pipeline_total_ms:.2f} ms</div>
        <div class='metric'><strong>Avg Detection/Frame</strong><br>{avg_detection_per_frame:.2f}</div>
        <div class='metric'><strong>YOLO Inference</strong><br>{avg_yolo_ms:.2f} ms</div>
        <div class='metric'><strong>Global Flow</strong><br>{avg_global_flow_ms:.2f} ms</div>
        <div class='metric'><strong>Object Flow</strong><br>{avg_object_flow_ms:.2f} ms</div>
        <div class='metric'><strong>Lane Segmentation</strong><br>{avg_lane_ms:.2f} ms</div>
        <div class='metric'><strong>Risk Fusion</strong><br>{avg_risk_ms:.2f} ms</div>
        <div class='metric'><strong>Scene Metrics</strong><br>{avg_scene_ms:.2f} ms</div>
        <div class='metric'><strong>Annotation Render</strong><br>{avg_annotation_ms:.2f} ms</div>
      </div>
    </div>

    <!-- ── Output card with view-mode selector ── -->
    <div class='card'>
      <h2>Output</h2>

      <!-- Dropdown-style tab selector -->
      <label for='view-mode-select' style='margin-bottom:8px;'>Pilih Mode Tampilan</label>
      <select id='view-mode-select' style='max-width:320px; margin-bottom:16px;'>
        <option value='video'>🎬 Output Video (Hasil Proses)</option>
        <option value='camera'>📷 Live Camera Stream</option>
        <option value='realsense'>🎯 RealSense Realtime Processing</option>
      </select>

      <!-- Panel: output video -->
      <div id='panel-video' class='view-panel active'>
        {f"<div class='notice'>{escaped_message}</div>" if message else "<p class='muted'>Belum ada proses terbaru.</p>"}
        {elapsed_text}
        {video_html if video_html else ""}
      </div>

      <!-- Panel: live camera -->
      <div id='panel-camera' class='view-panel'>
        <p>
          <strong>Live Camera Feed</strong>
          <span id='cam-badge' class='cam-badge'>● OFFLINE</span>
        </p>
        <p class='muted' style='margin-top:0; font-size:0.9rem;'>
          Stream langsung dari kamera yang terhubung ke server.
          Pastikan kamera tersedia dan <code>opencv-python</code> terinstall.
        </p>
        <!-- MJPEG displayed via <img> — works in all browsers including Chrome -->
        <img
          id='cam-feed'
          class='live-feed'
          src=''
          alt='Camera stream tidak aktif'
          width='840'
          style='background:#111; min-height:200px; display:block;'
        >
        <div style='margin-top:12px; display:flex; gap:10px;'>
          <button type='button' id='btn-start-cam'>▶ Mulai Stream</button>
          <button type='button' id='btn-stop-cam' style='background:#dc2626;' disabled>■ Stop Stream</button>
        </div>
      </div>

      <!-- Panel: realtime processed stream from RealSense -->
      <div id='panel-realsense' class='view-panel'>
        <p>
          <strong>RealSense Realtime Processed Feed</strong>
          <span id='rs-badge' class='cam-badge'>● OFFLINE</span>
        </p>
        <p class='muted' style='margin-top:0; font-size:0.9rem;'>
          Menangkap frame color + depth dari kamera stereo RealSense,
          lalu memproses deteksi, depth estimation, optical flow, dan risk secara near real-time.
        </p>
        <img
          id='rs-feed'
          class='live-feed'
          src=''
          alt='RealSense stream tidak aktif'
          width='840'
          style='background:#111; min-height:200px; display:block;'
        >
        <div style='margin-top:12px; display:flex; gap:10px;'>
          <button type='button' id='btn-start-rs'>▶ Mulai Realtime</button>
          <button type='button' id='btn-stop-rs' style='background:#dc2626;' disabled>■ Stop Realtime</button>
        </div>
      </div>
    </div>

  </div><!-- /container -->

  <script>
    // ── Form submit: show loading ────────────────────────────────────────────
    const form      = document.getElementById('process-form');
    const button    = document.getElementById('process-button');
    const indicator = document.getElementById('loading-indicator');
    form.addEventListener('submit', () => {{
      button.disabled = true;
      indicator.classList.add('active');
    }});

    // ── View-mode selector ───────────────────────────────────────────────────
    const modeSelect   = document.getElementById('view-mode-select');
    const panelVideo   = document.getElementById('panel-video');
    const panelCamera  = document.getElementById('panel-camera');
    const panelRs      = document.getElementById('panel-realsense');

    function switchPanel(mode) {{
      panelVideo.classList.toggle('active',  mode === 'video');
      panelCamera.classList.toggle('active', mode === 'camera');
      panelRs.classList.toggle('active', mode === 'realsense');
      // Stop stream when leaving camera panel
      if (mode !== 'camera') stopCam();
      if (mode !== 'realsense') stopRs();
    }}

    modeSelect.addEventListener('change', () => switchPanel(modeSelect.value));

    // ── Camera stream controls ───────────────────────────────────────────────
    const camFeed    = document.getElementById('cam-feed');
    const camBadge   = document.getElementById('cam-badge');
    const btnStart   = document.getElementById('btn-start-cam');
    const btnStop    = document.getElementById('btn-stop-cam');

    function startCam() {{
      // Append timestamp to bust any cached "offline" state
      camFeed.src = '/camera/stream?t=' + Date.now();
      camFeed.onerror = () => {{
        camBadge.textContent = '● ERROR';
        camBadge.className   = 'cam-badge';
        btnStart.disabled    = false;
        btnStop.disabled     = true;
      }};
      camBadge.textContent = '● LIVE';
      camBadge.className   = 'cam-badge live';
      btnStart.disabled    = true;
      btnStop.disabled     = false;
    }}

    function stopCam() {{
      camFeed.src          = '';
      camBadge.textContent = '● OFFLINE';
      camBadge.className   = 'cam-badge';
      btnStart.disabled    = false;
      btnStop.disabled     = true;
    }}

    btnStart.addEventListener('click', startCam);
    btnStop.addEventListener('click',  stopCam);

    // ── RealSense realtime processed stream controls ───────────────────────
    const rsFeed    = document.getElementById('rs-feed');
    const rsBadge   = document.getElementById('rs-badge');
    const btnStartRs = document.getElementById('btn-start-rs');
    const btnStopRs  = document.getElementById('btn-stop-rs');

    function startRs() {{
      rsFeed.src = '/realtime/stream?t=' + Date.now();
      rsFeed.onerror = () => {{
        rsBadge.textContent = '● ERROR';
        rsBadge.className   = 'cam-badge';
        btnStartRs.disabled = false;
        btnStopRs.disabled  = true;
      }};
      rsBadge.textContent = '● LIVE';
      rsBadge.className   = 'cam-badge live';
      btnStartRs.disabled = true;
      btnStopRs.disabled  = false;
    }}

    function stopRs() {{
      rsFeed.src          = '';
      rsBadge.textContent = '● OFFLINE';
      rsBadge.className   = 'cam-badge';
      btnStartRs.disabled = false;
      btnStopRs.disabled  = true;
    }}

    btnStartRs.addEventListener('click', startRs);
    btnStopRs.addEventListener('click',  stopRs);
  </script>
</body>
</html>
"""


# ── Startup & routes ─────────────────────────────────────────────────────────

@app.on_event("startup")
def startup_event():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    init_database()


@app.get("/", response_class=HTMLResponse)
def index():
    return _render_page()


@app.post("/process", response_class=HTMLResponse)
async def process(
    source_type: str = Form(...),
    bag_path: str = Form(default=""),
    bag_file: UploadFile | None = File(default=None),
    file: UploadFile | None = File(default=None),
):
    source_type = (source_type or "").strip().lower()

    temp_upload_path: Path | None = None
    source_path: Path | None = None

    try:
        app_logger.info(
            "POST /process received | source_type=%s | bag_file=%s | file=%s | bag_path=%s",
            source_type,
            bool(bag_file and bag_file.filename),
            bool(file and file.filename),
            bag_path,
        )

        if source_type not in {"bag"}:
            return _render_page(message="Source type sementara hanya bag (opsi mp4 dinonaktifkan).")

        # Prefer uploaded BAG when provided, otherwise fall back to local path.
        if bag_file is not None and bag_file.filename:
            if not bag_file.filename.lower().endswith(".bag"):
                return _render_page(message="File BAG harus berekstensi .bag.")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".bag") as temp_file:
                shutil.copyfileobj(bag_file.file, temp_file)
                temp_upload_path = Path(temp_file.name)
            source_path = temp_upload_path
        elif file is not None and file.filename and file.filename.lower().endswith(".bag"):
            # Compatibility fallback for older cached UI that may still post to `file`.
            with tempfile.NamedTemporaryFile(delete=False, suffix=".bag") as temp_file:
                shutil.copyfileobj(file.file, temp_file)
                temp_upload_path = Path(temp_file.name)
            source_path = temp_upload_path
        else:
            source_path = Path(bag_path.strip())
            if not source_path.exists() or source_path.suffix.lower() != ".bag":
                return _render_page(message="Path BAG tidak valid atau upload file BAG terlebih dahulu.")
        # else:
        #     if file is None or not file.filename:
        #         return _render_page(message="Silakan upload file MP4.")
        #     if not file.filename.lower().endswith(".mp4"):
        #         return _render_page(message="File harus berekstensi .mp4.")
        #
        #     with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        #         shutil.copyfileobj(file.file, temp_file)
        #         temp_upload_path = Path(temp_file.name)
        #     source_path = temp_upload_path

        if source_path is None:
            return _render_page(message="Sumber BAG belum ditentukan.")

        app_logger.info("Starting pipeline for BAG source: %s", source_path)
        service = VideoService(YOLO_MODEL_PATH)

        start = time.time()
        output_video_path = service.process(str(source_path), source_type)
        elapsed = time.time() - start

        output_video_name = Path(output_video_path).name

        return _render_page(
            message="Processing selesai.",
            output_video_name=output_video_name,
            elapsed_seconds=elapsed,
        )

    except Exception as exc:
        app_logger.error("Processing failed: %s", exc, exc_info=True)
        return _render_page(message=f"Processing gagal: {escape(str(exc))}")
    finally:
        if temp_upload_path and temp_upload_path.exists():
            temp_upload_path.unlink(missing_ok=True)