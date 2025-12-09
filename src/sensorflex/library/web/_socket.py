"""A node library for WebSocket server."""

from __future__ import annotations

import json
import asyncio
import websockets

from typing import Any

from sensorflex.core.flow import Node, Port, Event, FutureOp, FutureState


class WebSocketServerNode(Node):
    # Configuration ports
    host: str
    port: int

    # Output ports
    last_message: Port[dict[str, Any]]

    message: Event[dict[str, Any]]

    _server_op: FutureOp[None]

    def __init__(
        self, host: str = "0.0.0.0", port: int = 8765, name: str | None = None
    ) -> None:
        super().__init__(name)

        # Default configuration; can be overridden by the graph
        self.host = host
        self.port = port

        # Outputs
        self.last_message = Port({})
        self.message = Event({})

        # Internal async machinery
        self._server_op: FutureOp[None] = FutureOp(self._run_server)
        _ = self._server_op.start()

        # Track connected clients (ServerConnection objects in websockets ≥ 12)
        self._clients: set[Any] = set()

    def forward(self) -> None:
        """
        Called each tick by the graph.

        Uses FutureOp to start/monitor the async WebSocket server.
        """

        # Step the underlying async task
        match self._server_op.step(restart=True):
            case FutureState.COMPLETED | FutureState.FAILED | FutureState.CANCELLED:
                self._server_op.reset()

    async def _handler(self, conn: Any) -> None:
        """
        websockets ≥ 12 handler signature: (connection) -> Awaitable[None]
        """
        self._clients.add(conn)
        try:
            async for raw_msg in conn:
                try:
                    msg = json.loads(raw_msg)
                except json.JSONDecodeError:
                    msg = {"type": "raw", "data": raw_msg}

                # This triggers graph_to_notify.on_port_change(...)
                self.last_message <<= msg
                self.message <<= msg  # Invoke a new pipeline
        finally:
            self._clients.discard(conn)

    async def _run_server(self) -> None:
        """
        Coroutine run by FutureOp.

        Starts a WebSocket server and keeps it alive until cancelled.
        """

        async with websockets.serve(self._handler, self.host, self.port):
            print(
                f"[{self.name}] WebRTC signaling server at ws://{self.host}:{self.port}"
            )
            # Stay alive forever; FutureOp.cancel() will cancel the task
            await asyncio.Future()

    # Optional: async broadcast helper, if you ever call it from inside the loop
    async def broadcast(self, message: dict[str, Any]) -> None:
        if not self._clients:
            return

        raw = json.dumps(message)
        await asyncio.gather(
            *[conn.send(raw) for conn in list(self._clients) if not conn.closed],
            return_exceptions=True,
        )

    def cancel(self) -> None:
        """Cancel the server task via node API."""
        if self._server_op is not None:
            self._server_op.cancel()
