# src/sensorflex/cli/_cli.py

import rerun as rr
import argparse
from sensorflex.core._service import start_main_service
from sensorflex.utils.logging import configure_default_logging


def main():
    parser = argparse.ArgumentParser("sensorflex")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--no-png", action="store_true")
    args = parser.parse_args()

    configure_default_logging()

    rr.init("SensorFlex Real-time Visualization", spawn=True)

    import asyncio

    asyncio.run(
        start_main_service(
            websocket_host=args.host,
            websocket_port=args.port,
            use_default_rerun_video_visualization=not args.no_video,
            use_default_png_uint8_visualization=not args.no_png,
        )
    )
