"""Data visualization library."""

from dataclasses import dataclass
from typing import ClassVar

import rerun as rr
from aiortc import MediaStreamTrack

from PIL import Image
import numpy as np
import io

from sensorflex.utils.logging import get_logger

from sensorflex.library.web._types import (
    BaseVideoTrackHandler,
    BaseChunkedDataChannelHandler,
    BaseChunkedDataHeader,
)

LOGGER = get_logger("_visualization")


# ---------------------------------------------------------------------------
# Concrete handlers: Rerun integration
# ---------------------------------------------------------------------------


class RerunVideoVisualizationHandler(BaseVideoTrackHandler):
    """
    Video track handler that logs frames to Rerun.

    'name' is used both as a conceptual identifier and as the Rerun entity path.
    """

    name: str = "@sensorflex/camera_rgb_webrtc_bgr24"

    def __init__(self) -> None:
        super().__init__()

    def on_frame_received(
        self,
        frame_idx: int,
        ts: float,
        img_bgr: np.ndarray,
        track: MediaStreamTrack,
    ) -> None:
        rr.set_time("sender_time", timestamp=ts)
        rr.set_time("frame_idx", sequence=frame_idx)
        rr.log(self.name, rr.Image(img_bgr, color_model="BGR"))


class RerunChunkedPngDataChannelHandler(BaseChunkedDataChannelHandler):
    """
    PNG-over-datachannel handler that reassembles chunked PNG frames and logs
    them to Rerun using 'name' as the entity path.

    It uses the default BaseChunkedDataHeader layout:
        INNER_STRUCT_FMT = "IHHI"
    and only specifies MAGIC and VERSION.
    """

    name: str = "@sensorflex/camera_rgb_png_uint8"

    @dataclass
    class ChunkHeader(BaseChunkedDataHeader):
        # Meta information
        VERSION: ClassVar[int] = 1

        # Custom information to override.
        # xxx = "xxx"

        # Modify STRUCT_FMT if there is new custom data structure.
        # STRUCT_FMT = ""

    def __init__(self) -> None:
        super().__init__()

    def on_frame_received(
        self,
        header: BaseChunkedDataHeader,
        frame_bytes: bytes,
    ) -> None:
        """
        Decode the reassembled PNG frame and log it to Rerun.
        """
        try:
            img = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
            img_rgb = np.array(img)
            img_bgr = img_rgb[..., ::-1]
            rr.set_time("frame_idx", sequence=header.frame_id)
            rr.log(self.name, rr.Image(img_bgr, color_model="BGR"))
        except Exception as e:
            LOGGER.error(
                "Error decoding reassembled PNG for frame %d on '%s': %s",
                header.frame_id,
                self.name,
                e,
            )
