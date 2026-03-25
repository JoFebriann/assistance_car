import cv2
from pathlib import Path
import shutil
import subprocess
from utils.logger import get_logger
from config.settings import VIDEO_CONFIG


class VideoBuilder:

    def __init__(self):
        self.logger = get_logger("VideoBuilder")

    def build(self, frame_dir: Path, output_path: Path, fps=None):

        if fps is None:
            fps = VIDEO_CONFIG["default_fps"]

        images = sorted(frame_dir.glob("frame_*.png"))

        if not images:
            raise RuntimeError("No frames found to build video.")

        first = cv2.imread(str(images[0]))
        if first is None:
            raise RuntimeError(f"Unable to read first frame: {images[0]}")
        height, width = first.shape[:2]

        output_path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter.fourcc(*VIDEO_CONFIG["codec"])
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (width, height)
        )

        if not writer.isOpened():
            raise RuntimeError("VideoWriter failed to open.")

        for img in images:
            frame = cv2.imread(str(img))
            if frame is None:
                self.logger.warning(f"Skipping unreadable frame: {img}")
                continue
            writer.write(frame)

        writer.release()

        self.logger.info(f"Video built successfully: {output_path}")
        return output_path

    def attach_audio(self, video_path: Path, audio_path: Path, output_path: Path):

        ffmpeg_cmd = VIDEO_CONFIG["ffmpeg_command"]
        if shutil.which(ffmpeg_cmd) is None:
            try:
                import imageio_ffmpeg
                ffmpeg_cmd = imageio_ffmpeg.get_ffmpeg_exe()
                self.logger.info(f"Using imageio-ffmpeg binary: {ffmpeg_cmd}")
            except Exception:
                self.logger.warning(
                    "FFmpeg was not found in PATH and imageio-ffmpeg fallback is unavailable, "
                    "returning silent video."
                )
                if output_path != video_path:
                    output_path.write_bytes(video_path.read_bytes())
                return output_path

        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            ffmpeg_cmd,
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-c:v",
            "copy",
            "-c:a",
            VIDEO_CONFIG["audio_codec"],
            "-b:a",
            VIDEO_CONFIG["audio_bitrate"],
            "-shortest",
            str(output_path),
        ]

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.logger.info(f"Muxed video+audio: {output_path}")
            return output_path
        except subprocess.CalledProcessError as exc:
            self.logger.warning(
                "FFmpeg mux failed, returning silent video. stderr=%s",
                exc.stderr.decode("utf-8", errors="ignore")
            )
            if output_path != video_path:
                output_path.write_bytes(video_path.read_bytes())
            return output_path