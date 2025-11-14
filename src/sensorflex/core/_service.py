"""Main service library."""

import asyncio
import json
import logging
from typing import (
    Any,
    Dict,
    List,
    Optional,
    cast,
)

import rerun as rr
from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    MediaStreamTrack,
    RTCRtpSender,
)
from av import VideoFrame
import websockets

from sensorflex.core._types import (
    BaseVideoTrackHandler,
    BaseDataChannelHandler,
)
from sensorflex.core._visualization import (
    RerunVideoVisualizationHandler,
    RerunChunkedPngDataChannelHandler,
)
from sensorflex.utils.logging import get_logger

LOGGER = get_logger("_service")


# ---------------------------------------------------------------------------
# RTCPeerConnection wrapper
# ---------------------------------------------------------------------------
class RtcPeer:
    """
    Wraps RTCPeerConnection and delegates track/datachannel events to
    provided handlers.

    - video_handlers: list of BaseVideoTrackHandler instances; each receives
      every decoded video frame.
    - datachannel_handlers: list of BaseDataChannelHandler instances.
      The peer will ask each handler if it can handle a given datachannel label.
    """

    def __init__(
        self,
        video_handlers: List[BaseVideoTrackHandler],
        datachannel_handlers: List[BaseDataChannelHandler],
    ) -> None:
        self.pc = RTCPeerConnection()
        self._video_handlers = video_handlers
        self._datachannel_handlers = datachannel_handlers
        self._video_tasks: set[asyncio.Task] = set()

        self._register_callbacks()

    # ---- callbacks registration ----

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
            if track.kind == "video" and self._video_handlers:
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
        def _on_datachannel(channel) -> None:
            LOGGER.info("New data channel: label=%s", channel.label)

            handler = next(
                (h for h in self._datachannel_handlers if h.can_handle(channel.label)),
                None,
            )

            if handler is None:
                LOGGER.warning(
                    "No handler registered for datachannel label '%s'", channel.label
                )
                return

            @channel.on("message")
            def _on_message(message) -> None:
                handler.handle_message(message)

    # ---- video consumption ----

    async def _consume_video(self, track: MediaStreamTrack) -> None:
        LOGGER.info("consume_video started for track: %s", track.kind)
        frame_idx = 0

        while True:
            frame = cast(VideoFrame, await track.recv())
            img_bgr = frame.to_ndarray(format="bgr24")

            ts = frame.time
            if ts is None:
                ts = frame_idx / 30.0  # fallback

            for handler in self._video_handlers:
                handler.on_frame_received(frame_idx, ts, img_bgr, track)

            frame_idx += 1

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

    async def close(self) -> None:
        LOGGER.info("Closing RTCPeerConnection")
        for task in list(self._video_tasks):
            task.cancel()
        await self.pc.close()


# ---------------------------------------------------------------------------
# WebSocketSession: signaling layer
# ---------------------------------------------------------------------------
class WebSocketSession:
    """
    Handles WebSocket signaling for a single client, and owns an RtcPeer
    configured with specific handlers.
    """

    def __init__(
        self,
        websocket: Any,
        video_handlers: List[BaseVideoTrackHandler],
        datachannel_handlers: List[BaseDataChannelHandler],
    ) -> None:
        self.websocket = websocket
        self.rtc_peer = RtcPeer(
            video_handlers=video_handlers,
            datachannel_handlers=datachannel_handlers,
        )

    async def run(self) -> None:
        try:
            async for message in self.websocket:
                await self._handle_message(message)
        except Exception:
            LOGGER.exception("Error in signaling handler")
        finally:
            await self.rtc_peer.close()

    async def _handle_message(self, message: str) -> None:
        data: Dict[str, Any] = json.loads(message)
        msg_type = data.get("type")
        LOGGER.info("Signaling message from client: %s", msg_type)

        if msg_type == "offer":
            await self._handle_offer(data)
        else:
            LOGGER.warning("Unknown signaling message type: %s", msg_type)

    async def _handle_offer(self, data: Dict[str, Any]) -> None:
        offer = RTCSessionDescription(sdp=data["sdp"], type="offer")
        await self.rtc_peer.set_remote_description(offer)

        answer = await self.rtc_peer.create_and_set_answer()
        await self.rtc_peer.wait_for_ice_gathering_complete()

        LOGGER.info("Sending answer")
        await self.websocket.send(
            json.dumps(
                {
                    "type": answer.type,
                    "sdp": answer.sdp,
                }
            )
        )


# ---------------------------------------------------------------------------
# Server startup helpers
# ---------------------------------------------------------------------------
async def start_main_service(
    websocket_host: str = "0.0.0.0",
    websocket_port: int = 8765,
    *,
    video_track_handlers: Optional[List[BaseVideoTrackHandler]] = None,
    datachannel_handlers: Optional[List[BaseDataChannelHandler]] = None,
) -> None:
    """
    Start the WebSocket signaling server.

    Intended to be called from outside; default behavior is configurable:

    - use_default_rerun_video_visualization:
        If True, attach an internal RerunVideoVisualizationHandler that logs
        incoming video frames to Rerun under '@sensorflex/camera_rgb_webrtc_bgr24'.

    - use_default_png_uint8_visualization:
        If True, attach an internal RerunChunkedPngDataChannelHandler that expects a
        PNG-uint8 datachannel with label '@sensorflex/camera_rgb_png_uint8'
        and logs it to Rerun under that name.

    - extra_video_handlers:
        Optional additional video handlers (e.g. user-defined subclasses).

    - extra_datachannel_handlers:
        Optional additional datachannel handlers (e.g. depth, pose, etc.).
    """

    async def _handle_client(websocket):
        LOGGER.info(
            "New WebSocket signaling connection from %s", websocket.remote_address
        )

        video_handler_list: List[BaseVideoTrackHandler] = []
        datachannel_handler_list: List[BaseDataChannelHandler] = []

        # Extra handlers provided by caller (third-party / framework code)
        if video_track_handlers:
            video_handler_list.extend(video_track_handlers)

        if datachannel_handlers:
            datachannel_handler_list.extend(datachannel_handlers)

        session = WebSocketSession(
            websocket=websocket,
            video_handlers=video_handler_list,
            datachannel_handlers=datachannel_handler_list,
        )
        await session.run()

    async with websockets.serve(_handle_client, websocket_host, websocket_port):
        LOGGER.info(
            "WebSocket signaling server listening on ws://%s:%d",
            websocket_host,
            websocket_port,
        )
        await asyncio.Future()  # run forever


async def start_main_service_with_visualization(
    websocket_host: str = "0.0.0.0",
    websocket_port: int = 8765,
    *,
    use_default_rerun_video_visualization: bool = True,
    use_default_png_uint8_visualization: bool = True,
    extra_video_handlers: Optional[List[BaseVideoTrackHandler]] = None,
    extra_datachannel_handlers: Optional[List[BaseDataChannelHandler]] = None,
) -> None:
    """
    Start the WebSocket signaling server.

    Intended to be called from outside; default behavior is configurable:

    - use_default_rerun_video_visualization:
        If True, attach an internal RerunVideoVisualizationHandler that logs
        incoming video frames to Rerun under '@sensorflex/camera_rgb_webrtc_bgr24'.

    - use_default_png_uint8_visualization:
        If True, attach an internal RerunChunkedPngDataChannelHandler that expects a
        PNG-uint8 datachannel with label '@sensorflex/camera_rgb_png_uint8'
        and logs it to Rerun under that name.

    - extra_video_handlers:
        Optional additional video handlers (e.g. user-defined subclasses).

    - extra_datachannel_handlers:
        Optional additional datachannel handlers (e.g. depth, pose, etc.).
    """
    rr.init("SensorFlex Real-time Visualization", spawn=True)

    async def _handle_client(websocket):
        LOGGER.info(
            "New WebSocket signaling connection from %s", websocket.remote_address
        )

        video_handlers: List[BaseVideoTrackHandler] = []
        datachannel_handlers: List[BaseDataChannelHandler] = []

        # Default internal handlers
        if use_default_rerun_video_visualization:
            video_handlers.append(RerunVideoVisualizationHandler())

        if use_default_png_uint8_visualization:
            datachannel_handlers.append(RerunChunkedPngDataChannelHandler())

        # Extra handlers provided by caller (third-party / framework code)
        if extra_video_handlers:
            video_handlers.extend(extra_video_handlers)

        if extra_datachannel_handlers:
            datachannel_handlers.extend(extra_datachannel_handlers)

        session = WebSocketSession(
            websocket=websocket,
            video_handlers=video_handlers,
            datachannel_handlers=datachannel_handlers,
        )
        await session.run()

    async with websockets.serve(_handle_client, websocket_host, websocket_port):
        LOGGER.info(
            "WebSocket signaling server listening on ws://%s:%d",
            websocket_host,
            websocket_port,
        )
        await asyncio.Future()  # run forever


# ---------------------------------------------------------------------------
# Example entry point (can be omitted when integrated into a framework)
# ---------------------------------------------------------------------------
async def main() -> None:
    logging.basicConfig(level=logging.INFO)

    rr.init("webrtc_rerun_receiver", spawn=True)

    caps = RTCRtpSender.getCapabilities("video")
    LOGGER.info("Available video codecs: %s", [c.mimeType for c in caps.codecs])

    await start_main_service_with_visualization(
        websocket_host="0.0.0.0",
        websocket_port=8765,
        use_default_rerun_video_visualization=True,
        use_default_png_uint8_visualization=True,
        # You can pass extra handlers here if you want:
        # extra_video_handlers=[MyCustomVideoHandler()],
        # extra_datachannel_handlers=[MyCustomChunkedHandler()],
    )


if __name__ == "__main__":
    asyncio.run(main())
