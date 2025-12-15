"""A node library for WebRTC signaling using WebSocket."""

from __future__ import annotations

import cv2
import asyncio
import numpy as np
from typing import Any


from sensorflex.core.runtime import Node, Port, FutureOp


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
        self._webcam_top: FutureOp[None] = FutureOp(self._run_server)
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

    async def _read_cap(self):
        ok, frame = await asyncio.to_thread(self.cap.read)
        return ok, frame

    async def _run_server(self) -> None:
        """
        Coroutine run by FutureOp.

        Starts a WebSocket server and keeps it alive until cancelled.
        """

        while True:
            _, frame = await self._read_cap()
            self.o_frame <<= frame

    def cancel(self) -> None:
        """Cancel the server task via node API."""
        # TODO: this is not useable.
        if self._webcam_top is not None:
            self._webcam_top.cancel()


class RandImgNode(Node):
    o_frame: Port[cv2.typing.MatLike]

    _webcam_top: FutureOp[None]

    def __init__(self, device_index: int = 0, name: str | None = None) -> None:
        super().__init__(name)

        # Default configuration; can be overridden by the graph
        self.cap = cv2.VideoCapture(device_index)

        # Outputs
        self.o_frame = Port(None)

        # Internal async machinery
        self._webcam_top: FutureOp[None] = FutureOp(self._run_server)
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

    async def _run_server(self) -> None:
        """
        Coroutine run by FutureOp.

        Starts a WebSocket server and keeps it alive until cancelled.
        """

        while True:
            data_array = np.random.randint(0, 256, size=(1080, 1920, 3), dtype=np.uint8)
            self.o_frame <<= data_array

    def cancel(self) -> None:
        """Cancel the server task via node API."""
        # TODO: this is not useable.
        if self._webcam_top is not None:
            self._webcam_top.cancel()
