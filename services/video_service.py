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

        # 1️⃣ Setup FrameSaver for both output modes
        frame_saver_info = FrameSaver(f"{run_id}_information")
        frame_saver_driving = FrameSaver(f"{run_id}_driving")

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
                # Process once for analytics/database; render two visual modes from same results.
                result = self.pipeline.process_frame(frame_data)

                annotated_info = self.pipeline._draw_annotations(
                    rgb_image=frame_data.rgb_image,
                    calculations=result["detections"],
                    flow_stats=result["flow"],
                    scene_risk=result["scene_risk"],
                    scene_metrics=result.get("scene_metrics"),
                    lane_result=result.get("lane"),
                    perf_stats=result.get("performance"),
                    annotation_mode="information",
                )
                frame_saver_info.save(frame_data.frame_id, annotated_info)

                annotated_driving = self.pipeline._draw_annotations(
                    rgb_image=frame_data.rgb_image,
                    calculations=result["detections"],
                    flow_stats=result["flow"],
                    scene_risk=result["scene_risk"],
                    scene_metrics=result.get("scene_metrics"),
                    lane_result=result.get("lane"),
                    perf_stats=result.get("performance"),
                    annotation_mode="driving",
                )
                frame_saver_driving.save(frame_data.frame_id, annotated_driving)

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

        silent_video_info = self.output_dir / f"{run_id}_information_silent.mp4"
        output_video_info = self.output_dir / f"{run_id}_information.mp4"
        silent_video_driving = self.output_dir / f"{run_id}_driving_silent.mp4"
        output_video_driving = self.output_dir / f"{run_id}_driving.mp4"

        self.logger.info("Building final video from frames...")

        video_builder.build(
            frame_saver_info.get_frame_dir(),
            silent_video_info,
            fps=fps
        )
        video_builder.build(
            frame_saver_driving.get_frame_dir(),
            silent_video_driving,
            fps=fps
        )

        self.logger.info(f"Silent information video saved at: {silent_video_info}")
        self.logger.info(f"Silent driving video saved at: {silent_video_driving}")

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
        final_info_video = video_builder.attach_audio(
            video_path=silent_video_info,
            audio_path=wav_path,
            output_path=output_video_info
        )
        final_driving_video = video_builder.attach_audio(
            video_path=silent_video_driving,
            audio_path=wav_path,
            output_path=output_video_driving
        )

        if not VIDEO_CONFIG["keep_intermediate_silent_video"]:
            if silent_video_info.exists():
                silent_video_info.unlink()
            if silent_video_driving.exists():
                silent_video_driving.unlink()

        self.logger.info(f"Final information video saved at: {final_info_video}")
        self.logger.info(f"Final driving video saved at: {final_driving_video}")

        self.logger.info("==== FINISHED ====")

        return {
            "run_id": run_id,
            "information_video": str(final_info_video),
            "driving_video": str(final_driving_video),
            "audio_alert": str(wav_path),
        }