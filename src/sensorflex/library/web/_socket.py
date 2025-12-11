"""A node library for WebSocket server."""

from __future__ import annotations

import json
from uuid import uuid4, UUID
from dataclasses import dataclass
from websockets import serve
from websockets.asyncio.server import ServerConnection

from typing import Dict, Any

from sensorflex.core.runtime import Node, Port, FutureOp, FutureState


@dataclass
class WebSocketMessage:
    client_id: UUID
    payload: Dict[str, Any]


class WebSocketServerNode(Node):
    # Configuration ports
    host: str
    port: int

    # Output ports
    i_message: Port[WebSocketMessage]
    o_message: Port[WebSocketMessage]

    # Internal states
    _server_op: FutureOp[None]
    _clients: Dict[UUID, ServerConnection]

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        name: str | None = None,
    ) -> None:
        super().__init__(name)

        # Default configuration; can be overridden by the graph
        self.host = host
        self.port = port

        # Ports
        self.i_message = Port(None, self._send_message)
        self.o_message = Port(None)

        # Internal async machinery
        self._server_op: FutureOp[None] = FutureOp(self._run_server)
        _ = self._server_op.start()

        # Track connected clients (ServerConnection objects in websockets ≥ 12)
        self._clients = {}

    def forward(self) -> None:
        """
        Called each tick by the graph.

        Uses FutureOp to start/monitor the async WebSocket server.
        """

        # Step the underlying async task
        match self._server_op.step(restart=True):
            case FutureState.COMPLETED | FutureState.FAILED | FutureState.CANCELLED:
                self._server_op.reset()

    async def _run_server(self) -> None:
        """
        Coroutine run by FutureOp.

        Starts a WebSocket server and keeps it alive until cancelled.
        """

        async with serve(self._handle_client, self.host, self.port) as server:
            print(
                f"[{self.name}] WebRTC signaling server at ws://{self.host}:{self.port}"
            )
            # Stay alive forever; FutureOp.cancel() will cancel the task
            await server.serve_forever()

    async def _handle_client(self, conn: ServerConnection) -> None:
        """
        websockets ≥ 12 handler signature: (connection) -> Awaitable[None]
        """
        new_conn_id = uuid4()
        self._clients[new_conn_id] = conn

        try:
            async for raw_msg in conn:
                try:
                    msg = json.loads(raw_msg)
                except json.JSONDecodeError:
                    msg = {"type": "raw", "data": raw_msg}

                # This triggers graph_to_notify.on_port_change(...)
                self.o_message <<= WebSocketMessage(new_conn_id, msg)
        finally:
            del self._clients[new_conn_id]

    async def _send_message(self):
        msg = ~self.i_message

        if msg.client_id in self._clients:
            conn = self._clients[msg.client_id]
            await conn.send(json.dumps(msg.payload))
        else:
            print(f"Connection closed for {msg.client_id}")

    def cancel(self) -> None:
        """Cancel the server task via node API."""
        if self._server_op is not None:
            self._server_op.cancel()
