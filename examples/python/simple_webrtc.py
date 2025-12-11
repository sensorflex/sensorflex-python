"""A simple WebRTC server."""

import asyncio
from typing import Any
import numpy as np
from av import VideoFrame

from sensorflex import Graph, Node, Port
from sensorflex.library.web import WebSocketServerNode, WebRTCSessionNode
from sensorflex.library.visualization import init_rerun_context, RerunVideoVisNode
from sensorflex.utils.logging import configure_default_logging

configure_default_logging()


class PrintNode(Node):
    field: Port[Any]

    def __init__(self, name: str | None = None) -> None:
        super().__init__(name)
        self.field = Port(None)

    def forward(self):
        print(~self.field)


class VFXNode(Node):
    i_frame: Port[VideoFrame]
    o_frame: Port[VideoFrame]

    def __init__(self, name: str | None = None) -> None:
        super().__init__(name)
        self.i_frame = Port(None)
        self.o_frame = Port(None)

    def forward(self) -> None:
        frame = ~self.i_frame
        frame_rgb = frame.to_ndarray(format="rgb24")
        gray = (
            0.299 * frame_rgb[:, :, 0]
            + 0.587 * frame_rgb[:, :, 1]
            + 0.114 * frame_rgb[:, :, 2]
        ).astype(np.uint8)
        gray = np.stack([gray, gray, gray], axis=-1)

        out_frame = VideoFrame.from_ndarray(gray, format="rgb24")
        self.o_frame <<= out_frame


async def main():
    g = Graph()

    g += (n1 := WebSocketServerNode())
    g += (n2 := PrintNode())

    p_rtc = +n1.o_message
    p_rtc += (n_rtc := WebRTCSessionNode())
    p_rtc += n1.o_message >> n_rtc.i_message

    p_msg = +n_rtc.o_message
    p_msg += n1 + (n_rtc.o_message >> n1.i_message)
    p_msg += n2 + (n_rtc.o_message >> n2.field)

    p_data = +n_rtc.o_data
    p_data += n2 + (n_rtc.o_data >> n2.field)

    p_frame = +n_rtc.o_frame
    p_frame += (n_vfx := VFXNode())
    p_frame += n_rtc.o_frame >> n_vfx.i_frame
    p_frame += n_rtc + (n_vfx.o_frame >> n_rtc.i_frame)

    p_frame += (n_vis := RerunVideoVisNode())
    p_frame += n_vfx.o_frame >> n_vis.i_frame

    await g.wait_forever()


if __name__ == "__main__":
    init_rerun_context("WebRTC")
    asyncio.run(main())
