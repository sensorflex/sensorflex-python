"""A node library for WebRTC signaling using WebSocket."""

from __future__ import annotations

import cv2
import asyncio
import numpy as np
from typing import Any
from numpy.typing import NDArray

from sensorflex.core.runtime import Node, Port, FutureOp


def jpeg_encode_bgr(img_bgr: NDArray, quality: int = 90) -> NDArray:
    """
    img_bgr: uint8 NumPy array in BGR order (OpenCV default), shape (H,W,3) or (H,W)
    returns: JPEG bytes
    """
    if img_bgr.dtype != np.uint8:
        raise ValueError("img_bgr must be uint8")

    ok, buf = cv2.imencode(
        ".jpg",
        img_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
    )
    if not ok:
        raise RuntimeError("cv2.imencode failed")

    return buf


def png_encode(img: np.ndarray, compression: int = 3) -> NDArray:
    """
    img: uint8 NumPy array
         - (H, W) grayscale
         - (H, W, 3) BGR (OpenCV convention)
         - (H, W, 4) BGRA
    compression: 0 (none, fastest) → 9 (max, smallest)
    returns: PNG bytes
    """
    if img.dtype != np.uint8:
        raise ValueError("img must be uint8")

    ok, buf = cv2.imencode(
        ".png",
        img,
        [int(cv2.IMWRITE_PNG_COMPRESSION), int(compression)],
    )
    if not ok:
        raise RuntimeError("cv2.imencode failed")

    return buf


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
            t = frame
            frame = jpeg_encode_bgr(frame)
            print(t.size, frame.size, frame.size / t.size)
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
