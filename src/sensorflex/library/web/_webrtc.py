"""A node library for WebRTC signaling using WebSocket."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID, uuid4
import asyncio
from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    MediaStreamTrack,
    RTCDataChannel,
    RTCIceCandidate,
    VideoStreamTrack,
)
from av import VideoFrame
from typing import cast, Any, Dict

from sensorflex.core.runtime import Port, Node
from sensorflex.library.web._socket import WebSocketMessage
from sensorflex.utils.logging import get_logger

LOGGER = get_logger("_service")


class EchoVideoTrack(VideoStreamTrack):
    """
    A VideoStreamTrack that pulls frames from a queue and sends them back
    to the peer. You can feed this queue from your incoming video consumer
    or from any other processing pipeline.
    """

    def __init__(self, frame_queue: asyncio.Queue[VideoFrame]) -> None:
        super().__init__()
        self._queue = frame_queue

    async def recv(self) -> VideoFrame:
        # Get *at least* one frame
        frame = await self._queue.get()

        # Drain the queue so we always keep the most recent frame
        while not self._queue.empty():
            try:
                frame = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # Re-timestamp to keep aiortc happy
        pts, time_base = await self.next_timestamp()
        frame.pts = pts
        frame.time_base = time_base
        return frame


@dataclass
class DataChannelMessage:
    client_id: UUID
    payload: Dict[str, Any]


class WebRTCSessionNode(Node):
    """
    Handles WebSocket signaling for a single client, and owns an RtcPeer
    configured with specific handlers.

    This object is intentionally *per-connection*. Reconnects should be handled
    by creating a new WebSocketSession and (optionally) re-attaching any
    higher-level state in your application code.
    """

    i_message: Port[WebSocketMessage]
    i_frame: Port[VideoFrame]

    o_message: Port[WebSocketMessage]
    o_data: Port[DataChannelMessage]
    o_frame: Port[VideoFrame]

    _frame_queue: asyncio.Queue[VideoFrame]
    _channels: Dict[UUID, RTCDataChannel]

    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.i_message = Port(None, self.run)
        self.i_frame = Port(None, self.send_frame)

        self.o_message = Port(None)
        self.o_data = Port(None)
        self.o_frame = Port(None)

        self.pc = RTCPeerConnection()
        self._video_tasks: set[asyncio.Task] = set()
        self._channels = {}

        self._frame_queue = asyncio.Queue[VideoFrame](maxsize=1)
        self._test_track = EchoVideoTrack(self._frame_queue)
        self.pc.addTrack(self._test_track)

        self._register_callbacks()

    def forward(self) -> None:
        return super().forward()

    async def run(self) -> None:
        """Decode JSON and delegate to typed handler."""
        message = ~self.i_message

        client_id = message.client_id
        payload = message.payload

        msg_type = payload.get("type")
        LOGGER.info("Signaling message from client: %s", msg_type)

        match msg_type:
            case "offer":
                await self._handle_offer(client_id, payload)
            case "candidate":
                await self._handle_candidate(client_id, payload)
            case _:
                LOGGER.warning("Unknown signaling message type: %s", msg_type)

    async def _handle_offer(self, client_id: UUID, data: Dict[str, Any]) -> None:
        sdp = data.get("sdp")
        if not isinstance(sdp, str):
            LOGGER.exception("Invalid offer")
            return

        try:
            offer = RTCSessionDescription(sdp=sdp, type="offer")
            await self.set_remote_description(offer)

            answer = await self.create_and_set_answer()
            await self.wait_for_ice_gathering_complete()
        except Exception:
            LOGGER.exception("Failed to handle offer")
            return

        LOGGER.info("Sending answer")
        self.o_message <<= WebSocketMessage(
            client_id=client_id,
            payload={
                "type": answer.type,
                "sdp": answer.sdp,
            },
        )

    async def _handle_candidate(self, client_id: UUID, data: Dict[str, Any]) -> None:
        ice = data.get("candidate")

        # Browser sometimes sends `null` / None to mark end-of-candidates
        if ice is None:
            LOGGER.info("End-of-candidates from %s", client_id)
            # With current aiortc versions it's fine to just ignore this;
            # if you *really* want to signal end-of-candidates you can
            # call addIceCandidate(None), but some versions treat that oddly.
            return

        if not isinstance(ice, dict):
            LOGGER.warning("Invalid ICE candidate payload (not a dict): %r", ice)
            return

        LOGGER.debug("Raw ICE candidate from %s: %r", client_id, ice)

        cand = self._parse_ice_candidate(ice)
        if cand is None:
            LOGGER.info(
                "Parsed ICE candidate is None (end-of-candidates or invalid), ignore."
            )
            return

        try:
            await self.pc.addIceCandidate(cand)
            LOGGER.info("Added ICE candidate from %s", client_id)
        except Exception:
            LOGGER.exception("Failed to add ICE candidate: %r", cand)

    def _register_callbacks(self) -> None:
        @self.pc.on("connectionstatechange")
        def _on_connectionstatechange() -> None:
            LOGGER.info("PC connection state: %s", self.pc.connectionState)

        @self.pc.on("iceconnectionstatechange")
        def _on_iceconnectionstatechange() -> None:
            LOGGER.info("PC ICE connection state: %s", self.pc.iceConnectionState)

        @self.pc.on("track")
        async def _on_track(track: MediaStreamTrack) -> None:
            LOGGER.info("Got new track: kind=%s", track.kind)
            if track.kind == "video":
                task = asyncio.create_task(self._consume_video(track))
                self._video_tasks.add(task)

                def _done_callback(t: asyncio.Task) -> None:
                    self._video_tasks.discard(t)
                    if t.cancelled():
                        return
                    exc = t.exception()
                    if exc:
                        LOGGER.error("Video consumer task error: %s", exc)

                task.add_done_callback(_done_callback)

        @self.pc.on("datachannel")
        def _on_datachannel(channel: RTCDataChannel) -> None:
            LOGGER.info("New data channel: label=%s", channel.label)
            channel_id = uuid4()
            self._channels[channel_id] = channel

            @channel.on("message")
            def _on_message(message) -> None:
                self.o_data <<= DataChannelMessage(channel_id, message)

    # ---- video consumption ----
    async def _consume_video(self, track: MediaStreamTrack) -> None:
        LOGGER.info("consume_video started for track: %s", track.kind)

        while True:
            frame = cast(VideoFrame, await track.recv())
            self.o_frame <<= frame

    # ---- signaling helpers ----
    async def set_remote_description(self, offer: RTCSessionDescription) -> None:
        LOGGER.info("Setting remote description (offer)")
        await self.pc.setRemoteDescription(offer)

    async def create_and_set_answer(self) -> RTCSessionDescription:
        LOGGER.info("Creating answer")
        answer = await self.pc.createAnswer()
        await self.pc.setLocalDescription(answer)
        return cast(RTCSessionDescription, self.pc.localDescription)

    async def wait_for_ice_gathering_complete(self) -> None:
        while self.pc.iceGatheringState != "complete":
            LOGGER.info("Waiting ICE gathering… %s", self.pc.iceGatheringState)
            await asyncio.sleep(0.1)

    async def send_frame(self):
        frame = ~self.i_frame
        if frame is None:
            return

        # Always keep only the latest frame in the queue
        try:
            if self._frame_queue.full():
                _ = self._frame_queue.get_nowait()  # drop the stale frame

            self._frame_queue.put_nowait(frame)
        except asyncio.QueueFull:
            # If we somehow still get here, just drop this frame
            pass

    async def close(self) -> None:
        LOGGER.info("Closing RTCPeerConnection")
        for task in list(self._video_tasks):
            task.cancel()
        await self.pc.close()

    @staticmethod
    def _parse_ice_candidate(candidate_dict: Dict[str, Any]) -> RTCIceCandidate | None:
        """
        Convert a browser-style ICE candidate dict into an aiortc RTCIceCandidate.

        Expected shape, straight from JS:
        {
            "candidate": "candidate:... 1 udp ...",
            "sdpMid": "0" | "1" | ...,
            "sdpMLineIndex": 0 | 1 | ...
            ...
        }
        """
        try:
            cand_str = candidate_dict["candidate"]
        except KeyError:
            LOGGER.warning(
                "ICE candidate missing 'candidate' field: %r", candidate_dict
            )
            return None

        # Empty candidate string == end-of-candidates in WebRTC
        if not cand_str:
            return None

        parts = cand_str.split()
        # Basic sanity check: "candidate:<foundation> <component> <protocol> <priority> <ip> <port> typ <type> ..."
        if len(parts) < 8 or not parts[0].startswith("candidate:"):
            LOGGER.warning("Unexpected ICE candidate format: %r", cand_str)
            return None

        foundation = parts[0].split(":", 1)[1]
        component = int(parts[1])
        protocol = parts[2].lower()
        priority = int(parts[3])
        ip = parts[4]
        port = int(parts[5])
        cand_type = parts[7]

        try:
            return RTCIceCandidate(
                foundation=foundation,
                component=component,
                protocol=protocol,
                priority=priority,
                ip=ip,
                port=port,
                type=cand_type,
                sdpMid=candidate_dict.get("sdpMid"),
                sdpMLineIndex=int(candidate_dict.get("sdpMLineIndex", 0)),
            )
        except Exception as e:
            LOGGER.error(
                "Failed to construct RTCIceCandidate from %r: %s", candidate_dict, e
            )
            return None
