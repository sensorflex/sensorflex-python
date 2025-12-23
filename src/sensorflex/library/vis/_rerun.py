"""Data visualization library."""

import rerun as rr
from numpy.typing import NDArray

from sensorflex.core.runtime import Node, Port
from sensorflex.utils.logging import get_logger

LOGGER = get_logger("Visualization")


def init_rerun_context(application_id: str):
    rr.init(application_id, spawn=True)


class RerunVideoVisNode(Node):
    """
    Video track handler that logs frames to Rerun.
    """

    i_frame: Port[NDArray]

    _i: int

    def __init__(self) -> None:
        super().__init__()
        self._i = 0
        self.i_frame = Port(None)

    def forward(self) -> None:
        rgb = ~self.i_frame
        rr.set_time("frame_idx", sequence=self._i)
        rr.log(self.name, rr.Image(rgb, color_model="RGB"))
        self._i += 1
