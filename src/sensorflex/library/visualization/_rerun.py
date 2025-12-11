"""Data visualization library."""

import rerun as rr
from av import VideoFrame

from sensorflex.core.runtime import Node, Port
from sensorflex.utils.logging import get_logger


LOGGER = get_logger("_visualization")


# ---------------------------------------------------------------------------
# Concrete handlers: Rerun integration
# ---------------------------------------------------------------------------


def init_rerun_context(application_id: str):
    rr.init(application_id, spawn=True)


class RerunVideoVisNode(Node):
    """
    Video track handler that logs frames to Rerun.
    """

    i_frame: Port[VideoFrame]

    _i: int

    def __init__(self) -> None:
        super().__init__()
        self._i = 0
        self.i_frame = Port(None)

    def forward(self) -> None:
        rgb = (~self.i_frame).to_rgb().to_ndarray(channel_last=True)
        rr.set_time("frame_idx", sequence=self._i)
        rr.log(self.name, rr.Image(rgb, color_model="RGB"))
        self._i += 1


# class RerunChunkedPngDataVisNode(BaseChunkedDataChannelHandler):
#     """
#     PNG-over-datachannel handler that reassembles chunked PNG frames and logs
#     them to Rerun using 'name' as the entity path.

#     It uses the default BaseChunkedDataHeader layout:
#         INNER_STRUCT_FMT = "IHHI"
#     and only specifies MAGIC and VERSION.
#     """

#     name: str = "@sensorflex/camera_rgb_png_uint8"

#     @dataclass
#     class ChunkHeader(BaseChunkedDataHeader):
#         # Meta information
#         VERSION: ClassVar[int] = 1

#         # Custom information to override.
#         # xxx = "xxx"

#         # Modify STRUCT_FMT if there is new custom data structure.
#         # STRUCT_FMT = ""

#     def __init__(self) -> None:
#         super().__init__()

#     def on_frame_received(
#         self,
#         header: BaseChunkedDataHeader,
#         frame_bytes: bytes,
#     ) -> None:
#         """
#         Decode the reassembled PNG frame and log it to Rerun.
#         """
#         try:
#             img = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
#             img_rgb = np.array(img)
#             img_bgr = img_rgb[..., ::-1]
#             rr.set_time("frame_idx", sequence=header.frame_id)
#             rr.log(self.name, rr.Image(img_bgr, color_model="BGR"))
#         except Exception as e:
#             LOGGER.error(
#                 "Error decoding reassembled PNG for frame %d on '%s': %s",
#                 header.frame_id,
#                 self.name,
#                 e,
#             )
