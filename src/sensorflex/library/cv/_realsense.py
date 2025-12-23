"""A node library for using webcam."""

from __future__ import annotations

import asyncio
from typing import Any

import cv2

from sensorflex.core.runtime import FutureOp, Node, Port


class WebcamNode(Node):
    # Configuration ports
    cap: cv2.VideoCapture

    o_frame: Port[cv2.typing.MatLike]

    _webcam_top: FutureOp[None]

    def __init__(self, device_index: int = 0, name: str | None = None) -> None:
        super().__init__(name)

        # Default configuration; can be overridden by the graph
        self.cap = cv2.VideoCapture(device_index)

        # Outputs
        self.o_frame = Port(None)

        # Internal async machinery
        self._webcam_top: FutureOp[None] = FutureOp(self._run_coroutine)
        _ = self._webcam_top.start()

        # Track connected clients (ServerConnection objects in websockets ≥ 12)
        self._clients: set[Any] = set()

    def forward(self) -> None:
        """
        Called each tick by the graph.

        Uses FutureOp to start/monitor the async WebSocket server.
        """

        # Step the underlying async task
        # match self._webcam_top.step(start_new=True):
        #     case FutureState.COMPLETED | FutureState.FAILED | FutureState.CANCELLED:
        #         pass
        pass

    async def _read_cap_async(self):
        ok, frame = await asyncio.to_thread(self.cap.read)
        return ok, frame

    async def _run_coroutine(self) -> None:
        """
        Coroutine run by FutureOp.

        Starts a WebSocket server and keeps it alive until cancelled.
        """

        while True:
            _, frame = await self._read_cap_async()
            # t = frame
            # frame = jpeg_encode_bgr(frame)
            # print(t.size, frame.size, frame.size / t.size)
            self.o_frame <<= frame

    def cancel(self) -> None:
        """Cancel the server task via node API."""
        # TODO: this is not useable.
        if self._webcam_top is not None:
            self._webcam_top.cancel()
