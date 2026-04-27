from database.db import reset_database

from core.pipeline import PerceptionPipeline
from core.video.frame_saver import FrameSaver
from core.video.video_builder import VideoBuilder
from core.video.audio_alert import generate_alert_wav

from utils.logger import get_logger
from config.settings import OUTPUT_DIR, PROCESSING_CONFIG, VIDEO_CONFIG

from pathlib import Path
import time


class VideoService:

    def __init__(self, model_path):
        self.logger = get_logger("VideoService")
        self.pipeline = PerceptionPipeline(model_path)

        self.base_dir = Path(__file__).resolve().parent.parent
        self.output_dir = OUTPUT_DIR

    def process(self, source_path: str, source_type: str):

        self.logger.info("==== START PROCESSING ====")
        self.logger.info(f"Source: {source_path}")
        self.logger.info(f"Type: {source_type}")

        reset_database()
        self.pipeline.reset()
        run_started = time.perf_counter()

        run_id = Path(source_path).stem
        self.logger.info(f"Run ID: {run_id}")

        # 1️⃣ Setup FrameSaver
        frame_saver = FrameSaver(run_id)

        # 2️⃣ Select generator (keep instance reference to read fps after processing)
        if source_type == "bag":
            from services.bag_generator import BagFrameGenerator
            frame_gen = BagFrameGenerator(source_path)
        elif source_type == "mp4":
            from services.video_generator import VideoFrameGenerator
            frame_gen = VideoFrameGenerator(source_path)
        else:
            raise ValueError("source_type must be 'bag' or 'mp4'")

        generator = frame_gen.generate()

        alert_flags = []
        frame_count = 0

        # 3️⃣ Process frames
        for frame_data in generator:
            try:
                result = self.pipeline.process_frame(
                    frame_data,
                    frame_saver=frame_saver
                )

                alert_flags.append(result["alert"])
                frame_count += 1

            except Exception as e:
                self.logger.error(
                    f"Error on frame {frame_data.frame_id}: {e}",
                    exc_info=True
                )

        self.logger.info(f"Finished frame processing: {frame_count} frames")

        total_elapsed_s = time.perf_counter() - run_started
        throughput_fps = (frame_count / total_elapsed_s) if total_elapsed_s > 0 else 0.0
        self.logger.info(
            "Processing throughput: %.2f FPS over %.2fs",
            throughput_fps,
            total_elapsed_s,
        )

        # Read the actual FPS that was detected inside generate() during iteration
        fps = getattr(frame_gen, 'fps', PROCESSING_CONFIG["default_fps"])
        if not fps or fps <= 0:
            fps = PROCESSING_CONFIG["default_fps"]
        self.logger.info(f"Output FPS: {fps}")

        # 4️⃣ Build video from saved frames
        video_builder = VideoBuilder()

        silent_video_path = self.output_dir / f"{run_id}_silent.mp4"
        output_video_path = self.output_dir / f"{run_id}.mp4"

        self.logger.info("Building final video from frames...")

        video_builder.build(
            frame_saver.get_frame_dir(),
            silent_video_path,
            fps=fps
        )

        self.logger.info(f"Silent video saved at: {silent_video_path}")

        # 5️⃣ Generate alert audio (must use the same FPS as the video)
        wav_path = self.output_dir / f"{run_id}_alert.wav"

        self.logger.info("Generating alert audio...")

        generate_alert_wav(
            alert_flags=alert_flags,
            fps=fps,
            wav_path=wav_path
        )

        self.logger.info(f"Alert audio saved at: {wav_path}")

        # 6️⃣ Merge alert audio into video
        final_video_path = video_builder.attach_audio(
            video_path=silent_video_path,
            audio_path=wav_path,
            output_path=output_video_path
        )

        if not VIDEO_CONFIG["keep_intermediate_silent_video"] and silent_video_path.exists():
            silent_video_path.unlink()

        self.logger.info(f"Final video saved at: {final_video_path}")

        self.logger.info("==== FINISHED ====")

        return final_video_path