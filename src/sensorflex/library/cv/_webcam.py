"""A node library for WebRTC signaling using WebSocket."""

from __future__ import annotations

import asyncio
from enum import Enum, auto
from typing import Any, Callable, Dict

import cv2
import numpy as np
from numpy.typing import NDArray

from sensorflex.core.runtime import FutureOp, Node, Port


def jpeg_encode(img_bgr: NDArray, quality: int = 90) -> bytes:
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

    buf = buf.tobytes()

    return buf


def png_encode(img: np.ndarray, compression: int = 3) -> bytes:
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

    buf = buf.tobytes()

    return buf


def jpeg_decode(jpeg_bytes: bytes) -> NDArray:
    """
    jpeg_bytes: JPEG-encoded bytes
    returns: uint8 NumPy array in BGR order (H, W, 3)
    """
    buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    if img is None:
        raise RuntimeError("cv2.imdecode failed")

    return img


def png_decode(png_bytes: bytes, flags=cv2.IMREAD_UNCHANGED) -> NDArray:
    """
    png_bytes: PNG-encoded bytes
    flags:
      - cv2.IMREAD_UNCHANGED (preserve channels, incl. alpha)
      - cv2.IMREAD_COLOR
      - cv2.IMREAD_GRAYSCALE
    returns: uint8 NumPy array
    """
    buf = np.frombuffer(png_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, flags)

    if img is None:
        raise RuntimeError("cv2.imdecode failed")

    return img


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
            # t = frame
            # frame = jpeg_encode_bgr(frame)
            # print(t.size, frame.size, frame.size / t.size)
            self.o_frame <<= frame

    def cancel(self) -> None:
        """Cancel the server task via node API."""
        # TODO: this is not useable.
        if self._webcam_top is not None:
            self._webcam_top.cancel()


class ImageCodec(Enum):
    JPEG = auto()
    PNG = auto()


class ImageDecodeNode(Node):
    i_buf: Port[bytes]
    o_img: Port[NDArray]

    _codec: ImageCodec

    def __init__(self, codec: ImageCodec, name: str | None = None) -> None:
        super().__init__(name)
        self._codec = codec

        self.i_buf = Port(None)
        self.o_img = Port(None)

    def forward(self) -> None:
        codec_func_map: Dict[ImageCodec, Callable[[bytes], NDArray]] = {
            ImageCodec.JPEG: jpeg_decode,
            ImageCodec.PNG: png_decode,
        }

        if self._codec not in codec_func_map:
            raise ValueError("Unknown codec.")

        buf = ~self.i_buf
        img = codec_func_map[self._codec](buf)

        self.o_img <<= img


class ImageEncodeNode(Node):
    i_img: Port[NDArray]
    o_buf: Port[bytes]

    _codec: ImageCodec

    def __init__(self, codec: ImageCodec, name: str | None = None) -> None:
        super().__init__(name)
        self._codec = codec

        self.i_img = Port(None)
        self.o_buf = Port(None)

    def forward(self) -> None:
        codec_func_map: Dict[ImageCodec, Callable[[NDArray], bytes]] = {
            ImageCodec.JPEG: jpeg_encode,
            ImageCodec.PNG: png_encode,
        }

        if self._codec not in codec_func_map:
            raise ValueError("Unknown codec.")

        img = ~self.i_img
        buf = codec_func_map[self._codec](img)

        self.o_buf <<= buf


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
