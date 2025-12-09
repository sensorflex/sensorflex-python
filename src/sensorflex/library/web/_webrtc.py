"""A node library for WebRTC signaling using WebSocket."""

from __future__ import annotations

import asyncio
from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    MediaStreamTrack,
    RTCDataChannel,
)
import json
from av import VideoFrame
from typing import Any, Callable, Awaitable, Dict, cast

import websockets

from sensorflex.core.flow import Node, Port, Event, FutureOp, FutureState
from sensorflex.utils.logging import get_logger

LOGGER = get_logger("_service")


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


# # ---------------------------------------------------------------------------
# # RTCPeerConnection wrapper
# # ---------------------------------------------------------------------------
# class RtcPeer:
#     """
#     Wraps RTCPeerConnection and delegates track/datachannel events to
#     provided handlers.

#     - video_handlers: list of BaseVideoTrackHandler instances; each receives
#       every decoded video frame.
#     - datachannel_handlers: list of BaseDataChannelHandler instances.
#       The peer will ask each handler if it can handle a given datachannel label.
#     """

#     def __init__(
#         self,
#         video_handlers: List[BaseVideoTrackHandler],
#         datachannel_handlers: List[BaseDataChannelHandler],
#     ) -> None:
#         self.pc = RTCPeerConnection()
#         self._video_handlers = video_handlers
#         self._datachannel_handlers = datachannel_handlers
#         self._video_tasks: set[asyncio.Task] = set()

#         self._register_callbacks()

#     # ---- callbacks registration ----

#     def _register_callbacks(self) -> None:
#         @self.pc.on("connectionstatechange")
#         def _on_connectionstatechange() -> None:
#             LOGGER.info("PC connection state: %s", self.pc.connectionState)

#         @self.pc.on("iceconnectionstatechange")
#         def _on_iceconnectionstatechange() -> None:
#             LOGGER.info("PC ICE connection state: %s", self.pc.iceConnectionState)

#         @self.pc.on("track")
#         async def _on_track(track: MediaStreamTrack) -> None:
#             LOGGER.info("Got new track: kind=%s", track.kind)
#             if track.kind == "video" and self._video_handlers:
#                 task = asyncio.create_task(self._consume_video(track))
#                 self._video_tasks.add(task)

#                 def _done_callback(t: asyncio.Task) -> None:
#                     self._video_tasks.discard(t)
#                     if t.cancelled():
#                         return
#                     exc = t.exception()
#                     if exc:
#                         LOGGER.error("Video consumer task error: %s", exc)

#                 task.add_done_callback(_done_callback)

#         @self.pc.on("datachannel")
#         def _on_datachannel(channel: RTCDataChannel) -> None:
#             LOGGER.info("New data channel: label=%s", channel.label)

#             handler = next(
#                 (h for h in self._datachannel_handlers if h.can_handle(channel.label)),
#                 None,
#             )

#             if handler is None:
#                 LOGGER.warning(
#                     "No handler registered for datachannel label '%s'", channel.label
#                 )
#                 return

#             # Register current channel for the handler.
#             handler.channel = channel

#             @channel.on("message")
#             def _on_message(message) -> None:
#                 handler.handle_message(message)

#     # ---- video consumption ----

#     async def _consume_video(self, track: MediaStreamTrack) -> None:
#         LOGGER.info("consume_video started for track: %s", track.kind)
#         frame_idx = 0

#         while True:
#             frame = cast(VideoFrame, await track.recv())
#             img_bgr = frame.to_ndarray(format="bgr24")

#             ts = frame.time
#             if ts is None:
#                 ts = frame_idx / 30.0  # fallback

#             for handler in self._video_handlers:
#                 handler.on_frame_received(frame_idx, ts, img_bgr, track)

#             frame_idx += 1

#     # ---- signaling helpers ----

#     async def set_remote_description(self, offer: RTCSessionDescription) -> None:
#         LOGGER.info("Setting remote description (offer)")
#         await self.pc.setRemoteDescription(offer)

#     async def create_and_set_answer(self) -> RTCSessionDescription:
#         LOGGER.info("Creating answer")
#         answer = await self.pc.createAnswer()
#         await self.pc.setLocalDescription(answer)
#         return cast(RTCSessionDescription, self.pc.localDescription)

#     async def wait_for_ice_gathering_complete(self) -> None:
#         while self.pc.iceGatheringState != "complete":
#             LOGGER.info("Waiting ICE gathering… %s", self.pc.iceGatheringState)
#             await asyncio.sleep(0.1)

#     async def close(self) -> None:
#         LOGGER.info("Closing RTCPeerConnection")
#         for task in list(self._video_tasks):
#             task.cancel()
#         await self.pc.close()


# class WebRTCMessageNode(Node):
#     """
#     Node that consumes signaling messages (from WebSocketServer.last_message)
#     and runs the WebRTC offer → answer handshake using an RtcPeer.

#     - `message` should be a dict with at least:
#         {
#             "type": "offer",
#             "sdp": "<offer SDP string>"
#         }
#     - `rtc_peer` is an RtcPeer instance that knows how to handle the offer.
#     - `send_json` is an async callable used to send responses (answer / errors)
#       back to the client (e.g., WebSocket send or broadcast).
#     """

#     message: Port[dict[str, Any]]

#     def __init__(
#         self,
#         rtc_peer: RtcPeer,
#         send_json: Callable[[Dict[str, Any]], Awaitable[None]],
#         name: str | None = None,
#     ) -> None:
#         super().__init__(name)

#         # Input: signaling message from WebSocket
#         self.message = Port({})

#         # Dependencies
#         self.rtc_peer = rtc_peer
#         self._send_json: Callable[[Dict[str, Any]], Awaitable[None]] = send_json

#         # Async machinery
#         self.__handle_offer_op: FutureOp[None] = FutureOp(self._handle_offer)

#     def forward(self) -> None:
#         """
#         Called each tick by the graph.

#         If the current message is an 'offer', step the async handler via FutureOp.
#         """
#         data = ~self.message
#         if not isinstance(data, dict):
#             return

#         msg_type = data.get("type")
#         if msg_type != "offer":
#             return

#         # Step the offer-handling coroutine
#         _ = self.__handle_offer_op.start()
#         # match state:
#         #     case FutureState.COMPLETED | FutureState.FAILED | FutureState.CANCELLED:
#         #         self.__handle_offer_op.reset()

#     async def _handle_offer(self) -> None:
#         """
#         Handle the current 'offer' message in `self.message` and send back an answer.
#         """
#         data = ~self.message

#         if not isinstance(data, dict):
#             await self._safe_send_error(
#                 code="invalid_message",
#                 message="Expected a dict message for WebRTC offer.",
#             )
#             return

#         sdp = data.get("sdp")
#         if not isinstance(sdp, str):
#             await self._safe_send_error(
#                 code="invalid_offer",
#                 message="Offer must contain a string 'sdp' field.",
#             )
#             return

#         try:
#             # 1. Set remote description from offer
#             offer = RTCSessionDescription(sdp=sdp, type="offer")
#             await self.rtc_peer.set_remote_description(offer)

#             # 2. Create local answer and set it
#             answer = await self.rtc_peer.create_and_set_answer()

#             # 3. Wait for ICE gathering to complete so SDP is final
#             await self.rtc_peer.wait_for_ice_gathering_complete()

#         except Exception as exc:
#             await self._safe_send_error(
#                 code="offer_handling_failed",
#                 message=f"Failed to handle WebRTC offer: {exc}",
#             )
#             return

#         # 4. Send answer back to client
#         await self._safe_send_json(
#             {
#                 "type": answer.type,
#                 "sdp": answer.sdp,
#             }
#         )

#     async def _safe_send_json(self, payload: Dict[str, Any]) -> None:
#         """
#         Send a JSON-serializable dict back to the client, swallowing any send errors.
#         """
#         if self._send_json is None:
#             return

#         try:
#             await self._send_json(payload)
#         except Exception:
#             # Intentionally silent (no logger)
#             pass

#     async def _safe_send_error(self, code: str, message: str) -> None:
#         """
#         Helper to send an error response.
#         """
#         await self._safe_send_json(
#             {
#                 "type": "error",
#                 "code": code,
#                 "message": message,
#             }
#         )
