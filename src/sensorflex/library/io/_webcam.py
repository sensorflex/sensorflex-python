"""A node library for WebRTC signaling using WebSocket."""

from __future__ import annotations

import cv2
from typing import Any


from sensorflex.core.flow import Node, Port, ThreadOp, Action


class WebcamNode(Node):
    # Configuration ports
    cap: cv2.VideoCapture

    # Output ports
    last_frame: Port[cv2.typing.MatLike]

    frame: Action[cv2.typing.MatLike]

    _webcam_top: ThreadOp[None]

    def __init__(self, device_index: int = 0, name: str | None = None) -> None:
        super().__init__(name)

        # Default configuration; can be overridden by the graph
        self.cap = cv2.VideoCapture(device_index)

        # Outputs
        self.frame = Action(None)

        # Internal async machinery
        self._webcam_top: ThreadOp[None] = ThreadOp(self._run_server)
        _ = self._webcam_top.step(start_new=True)

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

    def _run_server(self) -> None:
        """
        Coroutine run by FutureOp.

        Starts a WebSocket server and keeps it alive until cancelled.
        """

        while True:
            _, frame = self.cap.read()
            # self.last_frame <<= frame
            self.frame <<= frame

    def cancel(self) -> None:
        """Cancel the server task via node API."""
        # TODO: this is not useable.
        if self._webcam_top is not None:
            self._webcam_top.cancel()
