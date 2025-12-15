"""A node library for WebSocket server."""

from __future__ import annotations

import json
import pstats
import asyncio
import cProfile
from asyncio import Lock
from uuid import uuid4, UUID
from dataclasses import dataclass
from websockets import serve

from websockets.asyncio.server import ServerConnection
from websockets.asyncio.client import connect
from websockets.asyncio.client import ClientConnection

from typing import Dict, Any, Optional, cast

from sensorflex.core.runtime import Node, Port, FutureOp, FutureState
from sensorflex.utils.logging import Perf


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

        async with serve(
            self._handle_client,
            self.host,
            self.port,
            max_size=None,
            write_limit=2**24,
            compression=None,
        ) as server:
            await server.serve_forever()

    async def _handle_client(self, conn: ServerConnection) -> None:
        """
        websockets ≥ 12 handler signature: (connection) -> Awaitable[None]
        """
        new_conn_id = uuid4()
        self._clients[new_conn_id] = conn

        try:
            async for raw_msg in conn:
                # try:
                #     msg = json.loads(raw_msg)
                # except json.JSONDecodeError:
                #     msg = {"type": "raw", "data": raw_msg}
                msg: bytes = cast(bytes, raw_msg)

                # This triggers graph_to_notify.on_port_change(...)
                self.o_message <<= WebSocketMessage(new_conn_id, {"data": msg})
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


class WebSocketClientNode(Node):
    uri: str  # e.g. "ws://localhost:8765"

    # Ports
    i_message: Port[Any]
    o_message: Port[Any]

    # Internal state
    _client_op: FutureOp[None]
    _conn: Optional[ClientConnection]
    _client_id: UUID

    _send_lock: Lock

    def __init__(
        self,
        uri: str = "ws://localhost:8765",
        name: str | None = None,
    ) -> None:
        super().__init__(name)

        # Default configuration; can be overridden by the graph
        self.uri = uri

        # This client's ID (used in WebSocketMessage.client_id)
        self._client_id = uuid4()

        # Ports
        # i_message writes trigger _send_message (async-safe via graph helper)
        self.i_message = Port(None, self._send_message)
        self.o_message = Port(None)

        # Internal async machinery
        self._conn = None
        self._client_op = FutureOp(self._run_client)
        _ = self._client_op.start()

        self._send_lock = Lock()

    def forward(self) -> None:
        """
        Called each tick by the graph.

        Uses FutureOp to start/monitor the async WebSocket client.
        If the client task completes or fails, it will be restarted
        on the next tick (reconnect behavior).
        """
        match self._client_op.step(restart=True):
            case FutureState.COMPLETED | FutureState.FAILED | FutureState.CANCELLED:
                self._client_op.reset()

    async def _run_client(self) -> None:
        """
        Coroutine run by FutureOp.

        Establishes a WebSocket connection and keeps reading messages
        until the connection is closed or the task is cancelled.
        """
        async with connect(
            self.uri, max_size=None, write_limit=2**24, compression=None
        ) as conn:
            self._conn = conn
            try:
                async for raw_msg in conn:
                    try:
                        msg = json.loads(raw_msg)
                    except json.JSONDecodeError:
                        msg = {"type": "raw", "data": raw_msg}

                    self.o_message <<= msg
            finally:
                self._conn = None

    async def _send_message(self) -> None:
        """
        Called when i_message is written to.

        Uses the current WebSocket connection (if available) to send
        the payload to the server.
        """
        msg = ~self.i_message
        if msg is None:
            return

        # Allow user to pass either WebSocketMessage or raw dict
        if isinstance(msg, WebSocketMessage):
            payload = msg.payload
        else:
            # Fallback: treat as raw payload dict
            payload = msg

        # print(self._conn, payload)
        if self._conn is not None:
            if isinstance(payload, bytes):
                data = payload
            else:
                data = json.dumps(payload)

            try:
                # with Perf("self._conn.send(data)"):
                # with cProfile.Profile() as pr:
                async with self._send_lock:
                    await self._conn.send(data)
                # pstats.Stats(pr).sort_stats("tottime").print_stats(30)

            except Exception as e:
                print("error", e)

        else:
            # You could swap this to a logger if you have one
            print("[WebSocketClientNode] No active connection to send message.")

    def cancel(self) -> None:
        """Cancel the client task via node API."""
        if self._client_op is not None:
            self._client_op.cancel()
