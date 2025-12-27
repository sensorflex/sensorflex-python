"""A node library for WebSocket server."""

from __future__ import annotations

from asyncio import Lock
from dataclasses import dataclass, field
from re import Pattern
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generic,
    Literal,
    Optional,
    Sequence,
    TypeVar,
)
from uuid import UUID, uuid4

from websockets import serve
from websockets.asyncio.client import ClientConnection, connect, process_exception
from websockets.asyncio.server import ServerConnection
from websockets.datastructures import HeadersLike
from websockets.extensions.base import ClientExtensionFactory, ServerExtensionFactory
from websockets.http11 import SERVER, USER_AGENT, Request, Response
from websockets.typing import Data, Origin, Subprotocol

from sensorflex.core.runtime import FutureOp, FutureState, Node, Port
from sensorflex.utils.logging import get_logger

logger = get_logger("WebSocket")


@dataclass(frozen=True, slots=True)
class WebSocketServerConfig:
    # WebSocket
    origins: Sequence[Origin | Pattern[str] | None] | None = None
    extensions: Sequence[ServerExtensionFactory] | None = None
    subprotocols: Sequence[Subprotocol] | None = None
    select_subprotocol: (
        Callable[
            [ServerConnection, Sequence[Subprotocol]],
            Subprotocol | None,
        ]
        | None
    ) = None
    compression: str | None = "deflate"

    # HTTP
    process_request: (
        Callable[
            [ServerConnection, Request],
            Awaitable[Response | None] | Response | None,
        ]
        | None
    ) = None
    process_response: (
        Callable[
            [ServerConnection, Request, Response],
            Awaitable[Response | None] | Response | None,
        ]
        | None
    ) = None
    server_header: str | None = SERVER

    # Timeouts
    open_timeout: float | None = 10
    ping_interval: float | None = 20
    ping_timeout: float | None = 20
    close_timeout: float | None = 10

    # Limits
    max_size: int | None = 2**20
    max_queue: int | None | tuple[int | None, int | None] = 16
    write_limit: int | tuple[int, int | None] = 2**15

    # Escape hatch
    create_connection: type[ServerConnection] | None = None

    # loop.create_server passthrough
    kwargs: dict[str, Any] = field(default_factory=dict)


T = TypeVar("T")


@dataclass
class WebSocketMessageEnvelope(Generic[T]):
    client_id: UUID
    payload: T


class WebSocketServerNode(Node):
    # Configuration ports
    host: str
    port: int

    # Output ports
    i_message: Port[WebSocketMessageEnvelope[Data]]
    o_message: Port[WebSocketMessageEnvelope[Data]]

    i_broadcast_message: Port[Data]

    # Internal states
    _config: WebSocketServerConfig
    _server_op: FutureOp[None]
    _clients: Dict[UUID, ServerConnection]

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        config: WebSocketServerConfig | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(name)

        # Default configuration; can be overridden by the graph
        self.host = host
        self.port = port

        # Ports
        self.i_message = Port(None, self._send_message)
        self.o_message = Port(None)

        self.i_broadcast_message = Port(None, self._broadcast_message)

        # Internal async machinery
        self._config = config or WebSocketServerConfig()
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

        # For Python >= 3.10
        # match self._server_op.step(restart=True):
        #     case FutureState.COMPLETED | FutureState.FAILED | FutureState.CANCELLED:
        #         self._server_op.reset()

        if (_ := self._server_op.step(restart=True)) in (
            FutureState.COMPLETED,
            FutureState.FAILED,
            FutureState.CANCELLED,
        ):
            self._server_op.reset()

    async def _run_server(self) -> None:
        """
        Coroutine run by FutureOp.

        Starts a WebSocket server and keeps it alive until cancelled.
        """

        c = self._config

        async with serve(
            self._handle_client,
            self.host,
            self.port,
            origins=c.origins,
            extensions=c.extensions,
            subprotocols=c.subprotocols,
            select_subprotocol=c.select_subprotocol,
            compression=c.compression,
            process_request=c.process_request,
            process_response=c.process_response,
            server_header=c.server_header,
            open_timeout=c.open_timeout,
            ping_interval=c.ping_interval,
            ping_timeout=c.ping_timeout,
            close_timeout=c.close_timeout,
            max_size=c.max_size,
            max_queue=c.max_queue,
            write_limit=c.write_limit,
            logger=logger,
            **c.kwargs,
        ) as server:
            await server.serve_forever()

    async def _handle_client(self, conn: ServerConnection) -> None:
        """
        websockets ≥ 12 handler signature: (connection) -> Awaitable[None]
        """
        new_conn_id = uuid4()
        self._clients[new_conn_id] = conn

        try:
            async for msg in conn:
                self.o_message <<= WebSocketMessageEnvelope(new_conn_id, msg)
        finally:
            del self._clients[new_conn_id]

    async def _send_message(self):
        msg = ~self.i_message

        if msg.client_id in self._clients:
            conn = self._clients[msg.client_id]
            await conn.send(msg.payload)
        else:
            logger.info(f"Connection closed for {msg.client_id}")

    async def _broadcast_message(self):
        msg = ~self.i_broadcast_message

        for conn in self._clients.values():
            await conn.send(msg)

    def cancel(self) -> None:
        """Cancel the server task via node API."""
        if self._server_op is not None:
            self._server_op.cancel()


@dataclass(frozen=True, slots=True)
class WebSocketClientConfig:
    # WebSocket
    origin: Origin | None = None
    extensions: Sequence[ClientExtensionFactory] | None = None
    subprotocols: Sequence[Subprotocol] | None = None
    compression: str | None = "deflate"
    # HTTP
    additional_headers: HeadersLike | None = None
    user_agent_header: str | None = USER_AGENT
    proxy: str | Literal[True] | None = True
    process_exception: Callable[[Exception], Exception | None] = process_exception
    # Timeouts
    open_timeout: float | None = 10
    ping_interval: float | None = 20
    ping_timeout: float | None = 20
    close_timeout: float | None = 10
    # Limits
    max_size: int | None = 2**20
    max_queue: int | None | tuple[int | None, int | None] = 16
    write_limit: int | tuple[int, int | None] = 2**15
    # Escape hatch
    create_connection: type[ClientConnection] | None = None
    # loop.create_connection kwargs passthrough
    kwargs: dict[str, Any] = field(default_factory=dict)


class WebSocketClientNode(Node):
    uri: str  # e.g. "ws://localhost:8765"

    # Ports
    i_message: Port[Data]
    o_message: Port[Data]

    # Internal state
    _client_op: FutureOp[None]
    _conn: Optional[ClientConnection]
    _client_id: UUID

    _send_lock: Lock

    def __init__(
        self,
        uri: str = "ws://localhost:8765",
        config: WebSocketClientConfig | None = None,
        # Other keyword arguments are passed to loop.create_connection
        name: str | None = None,
        **kwargs: Any,
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
        self._client_config = config or WebSocketClientConfig(kwargs=kwargs)
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
        if (_ := self._client_op.step(restart=True)) in (
            FutureState.COMPLETED,
            FutureState.FAILED,
            FutureState.CANCELLED,
        ):
            self._client_op.reset()

    async def _run_client(self) -> None:
        """
        Coroutine run by FutureOp.

        Establishes a WebSocket connection and keeps reading messages
        until the connection is closed or the task is cancelled.
        """
        c = self._client_config

        async with connect(
            self.uri,
            origin=c.origin,
            extensions=c.extensions,
            subprotocols=c.subprotocols,
            compression=c.compression,
            additional_headers=c.additional_headers,
            user_agent_header=c.user_agent_header,
            proxy=c.proxy,
            process_exception=c.process_exception,
            open_timeout=c.open_timeout,
            ping_interval=c.ping_interval,
            ping_timeout=c.ping_timeout,
            close_timeout=c.close_timeout,
            max_size=c.max_size,
            max_queue=c.max_queue,
            write_limit=c.write_limit,
            create_connection=c.create_connection,
            logger=logger,
            **c.kwargs,
        ) as conn:
            self._conn = conn
            try:
                async for raw_msg in conn:
                    self.o_message <<= raw_msg
            finally:
                self._conn = None

    async def _send_message(self) -> None:
        """
        Called when i_message is written to.

        Uses the current WebSocket connection (if available) to send
        the payload to the server.
        """
        payload = ~self.i_message

        if self._conn is not None:
            try:
                # with Perf("self._conn.send(data)"):
                # with cProfile.Profile() as pr:
                async with self._send_lock:
                    await self._conn.send(payload)
                # pstats.Stats(pr).sort_stats("tottime").print_stats(30)

            except Exception as e:
                logger.error(e)

        else:
            # You could swap this to a logger if you have one
            logger.error("No active server connection to send message.")

    def cancel(self) -> None:
        """Cancel the client task via node API."""
        if self._client_op is not None:
            self._client_op.cancel()


# TODO: probably implement edge op to make this a generalizable design.
class MessageDistributionNode(Node):
    i_message: Port[Data]
    o_str: Port[str]
    o_bytes: Port[bytes]

    def __init__(self, name: str | None = None) -> None:
        super().__init__(name)

        self.i_message = Port(None)
        self.o_str = Port(None)
        self.o_bytes = Port(None)

    def forward(self) -> None:
        msg = ~self.i_message

        if isinstance(msg, str):
            self.o_str <<= msg
        elif isinstance(msg, bytes):
            self.o_bytes <<= msg
        else:
            raise ValueError("Unknown message type.")
