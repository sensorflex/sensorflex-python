"""An example for simple program usage."""

import asyncio
import numpy as np
from aiortc import MediaStreamTrack

from sensorflex import start_main_service, BaseVideoTrackHandler


class VideoTrackingHandler(BaseVideoTrackHandler):
    """
    Video track handler that logs frames to Rerun.

    'name' is used both as a conceptual identifier and as the Rerun entity path.
    """

    name: str = "rgb_tracker"

    def __init__(self) -> None:
        super().__init__()

    def on_frame_received(
        self,
        frame_idx: int,
        ts: float,
        img_bgr: np.ndarray,
        track: MediaStreamTrack,
    ) -> None:
        print("New frame received", img_bgr.shape)


async def main():
    """Main function."""

    print("Start new Python framework server.")
    await start_main_service(video_track_handlers=[VideoTrackingHandler()])


if __name__ == "__main__":
    asyncio.run(main())
