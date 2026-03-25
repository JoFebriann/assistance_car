import argparse
from pathlib import Path

from services.video_service import VideoService
from database.db import init_database
from config.settings import MODEL_PATH


def main():

    parser = argparse.ArgumentParser(
        description="Assistance Car Backend CLI"
    )

    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to .bag or .mp4 file"
    )

    parser.add_argument(
        "--type",
        type=str,
        required=True,
        choices=["bag", "mp4"],
        help="Source type"
    )

    args = parser.parse_args()

    # Initialize DB
    init_database()

    print(f"Processing file: {args.source}")
    service = VideoService(MODEL_PATH)
    service.process(args.source, args.type)


if __name__ == "__main__":
    main()