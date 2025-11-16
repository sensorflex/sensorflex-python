"""Core types library."""

import struct
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Optional,
    Type,
    TypeVar,
    ClassVar,
)
from abc import ABC, abstractmethod

from aiortc import MediaStreamTrack, RTCDataChannel

import numpy as np
from sensorflex.utils.logging import get_logger

LOGGER = get_logger("_service")

H = TypeVar("H", bound="BaseDataHeader")


class BaseVideoTrackHandler(ABC):
    """
    Base class for video track handlers.

    Subclasses must define:

        name: str = "@sensorflex/..."

    which is used as an identifier / stream name (e.g., for Rerun entity path).
    """

    name: str  # must be defined by subclasses

    def __init__(self) -> None:
        if not getattr(self, "name", None):
            raise ValueError(
                f"{self.__class__.__name__} must define a non-empty 'name' class attribute"
            )

    @abstractmethod
    def on_frame_received(
        self,
        frame_idx: int,
        ts: float,
        img_bgr: np.ndarray,
        track: MediaStreamTrack,
    ) -> None:
        """
        Handle a decoded video frame from a MediaStreamTrack.

        - frame_idx: monotonically increasing index for this track.
        - ts: sender-side timestamp (seconds).
        - img_bgr: numpy array in BGR24 format.
        """
        raise NotImplementedError


@dataclass
class BaseDataHeader:
    """
    Base class for binary headers.

    Subclasses may define:
        - MAGIC: Optional[bytes]   (magic prefix; may be None)
        - VERSION: Optional[int]   (1-byte version; may be None)
        - STRUCT_FMT: Optional[str]  (full struct format for the dynamic fields)
        - BYTE_ORDER: str            (default '>')
        - INNER_STRUCT_FMT: Optional[str] (struct for fields, without byte order)

    If STRUCT_FMT is not provided, it will be built as:
        BYTE_ORDER + INNER_STRUCT_FMT

    The size() method computes the total header size in bytes, including:
        - len(MAGIC) if present
        - 1 byte for VERSION if present
        - struct.calcsize(get_struct_fmt())
    """

    # No instance fields here; concrete subclasses add them.
    VERSION: ClassVar[int] = 1

    # Either STRUCT_FMT or INNER_STRUCT_FMT must be defined by subclasses.
    STRUCT_FMT: ClassVar[str] = ""
    BYTE_ORDER: ClassVar[str] = ">"
    INNER_STRUCT_FMT: ClassVar[str] = ""

    @classmethod
    def get_struct_fmt(cls) -> str:
        return cls.BYTE_ORDER + cls.INNER_STRUCT_FMT + cls.STRUCT_FMT

    @classmethod
    def size(cls) -> int:
        version_size = 1 if cls.VERSION is not None else 0
        struct_fmt = cls.get_struct_fmt()
        struct_size = struct.calcsize(struct_fmt)
        return version_size + struct_size


class BaseDataChannelHandler(ABC):
    """
    Base class for data channel handlers.

    Subclasses must define a class attribute:

        name: str = "@sensorflex/..."

    which is used as the WebRTC datachannel label and also an identifier
    for logging/visualization (e.g., Rerun entity path).
    """

    name: str  # must be defined by subclasses
    channel: Optional[RTCDataChannel]

    def __init__(self) -> None:
        if not getattr(self, "name", None):
            raise ValueError(
                f"{self.__class__.__name__} must define a non-empty 'name' class attribute"
            )

    def can_handle(self, label: str) -> bool:
        """
        Decide whether this handler should handle a given datachannel label.
        Default behavior is exact string match, but third-party handlers can
        override this for pattern-based matching.
        """
        return label == self.name

    @abstractmethod
    def handle_message(self, message: Any) -> None:
        """
        Process an incoming message from the data channel.

        Implementations should assume `message` can be either `str` or `bytes`
        (or a bytes-like object) depending on how the sender uses the channel.
        """
        raise NotImplementedError

    def send_message(self, message: bytes) -> None:
        if self.channel is None:
            raise ValueError("Handler not registered with channel.")

        self.channel.send(message)

    # ---- generic header parsing helper ----
    @staticmethod
    def parse_struct_header(
        buf: memoryview,
        header_cls: Type[H],
    ) -> Optional[H]:
        """
        Generic helper to parse a fixed-layout header with optional magic and
        version.

        header_cls must be a subclass of BaseDataHeader and define:
            - MAGIC: Optional[bytes]
            - VERSION: Optional[int]
            - STRUCT_FMT or INNER_STRUCT_FMT (+ BYTE_ORDER)

        It may rely on BaseDataHeader.size() and get_struct_fmt().
        """
        if not issubclass(header_cls, BaseDataHeader):
            raise TypeError(
                f"parse_struct_header expects a subclass of BaseDataHeader, "
                f"got {header_cls.__name__}"
            )

        size = header_cls.size()
        if len(buf) < size:
            LOGGER.warning(
                "Buffer too small for header %s: needed=%d, got=%d",
                header_cls.__name__,
                size,
                len(buf),
            )
            return None

        offset = 0

        # Version check
        version_expected = header_cls.VERSION
        if version_expected is not None:
            version = buf[offset]
            if version != version_expected:
                LOGGER.warning(
                    "Unsupported header version %s (expected %s) in %s",
                    version,
                    version_expected,
                    header_cls.__name__,
                )
                return None
            offset += 1

        fmt = header_cls.get_struct_fmt()

        try:
            values = struct.unpack_from(fmt, buf, offset)
        except struct.error as e:
            LOGGER.error(
                "Error unpacking header struct for %s: %s",
                header_cls.__name__,
                e,
            )
            return None

        calc_size = offset + struct.calcsize(fmt)
        if size != calc_size:
            LOGGER.warning(
                "Header size mismatch for %s: declared=%d, computed=%d",
                header_cls.__name__,
                size,
                calc_size,
            )

        return header_cls(*values)  # type: ignore[call-arg]


@dataclass
class BaseChunkedDataHeader(BaseDataHeader):
    """
    Base header for chunked messages.

    Dynamic fields (per frame/chunk):
        - frame_id: int
        - chunk_idx: int
        - total_chunks: int
        - payload_len: int

    Default INNER_STRUCT_FMT handles exactly these fields:
        "IHHI"  (frame_id, chunk_idx, total_chunks, payload_len)
    """

    frame_id: int
    chunk_idx: int
    total_chunks: int
    payload_len: int

    # Default layout for standard chunk headers
    INNER_STRUCT_FMT: ClassVar[str] = "IHHI"


class BaseChunkedDataChannelHandler(BaseDataChannelHandler):
    """
    Base class for datachannel handlers that receive chunked frames.

    Subclasses must define:

        @dataclass
        class ChunkHeader(BaseChunkedDataHeader):
            # Optionally override:
            #   MAGIC, VERSION, INNER_STRUCT_FMT, BYTE_ORDER, STRUCT_FMT

    The base implementation will:
    - discover `ChunkHeader` via introspection on the subclass
    - parse headers using BaseDataChannelHandler.parse_struct_header
    - validate payload length
    - store chunks per frame_id
    - assemble when complete
    - call `on_chunk_received(header, payload_bytes)` for each chunk
    - call `on_frame_received(header, frame_bytes)` when a frame is complete
    """

    @dataclass
    class FrameBuffer:
        total_chunks: int
        chunks: Dict[int, bytes]
        received: int

    def __init__(self) -> None:
        super().__init__()
        # Introspect the subclass for ChunkHeader
        header_cls = getattr(self.__class__, "ChunkHeader", None)
        if header_cls is None:
            raise TypeError(
                f"{self.__class__.__name__} must define an inner `ChunkHeader` class "
                f"that subclasses BaseChunkedDataHeader"
            )
        if not issubclass(header_cls, BaseChunkedDataHeader):
            raise TypeError(
                f"{self.__class__.__name__}.ChunkHeader must be a subclass of "
                f"BaseChunkedDataHeader"
            )
        self._header_cls: Type[BaseChunkedDataHeader] = header_cls
        self._buffers: Dict[int, BaseChunkedDataChannelHandler.FrameBuffer] = {}

    # ---- hooks to override ----------------------------------------------

    def on_chunk_received(
        self,
        header: BaseChunkedDataHeader,
        chunk_payload: bytes,
    ) -> None:
        """
        Optional hook: called for each valid chunk payload.
        Default implementation does nothing.
        """
        pass

    @abstractmethod
    def on_frame_received(
        self,
        header: BaseChunkedDataHeader,
        frame_bytes: bytes,
    ) -> None:
        """
        Called when all chunks for a given frame have been received and
        concatenated into `frame_bytes`.
        """
        pass

    # ---- core implementation --------------------------------------------

    def handle_message(self, message: Any) -> None:
        # Text messages are ignored by default in chunked/binary handlers.
        if isinstance(message, str):
            LOGGER.debug(
                "Ignoring text message on chunked datachannel '%s': %s",
                self.name,
                message,
            )
            return

        buf = memoryview(message)
        header_cls = self._header_cls

        header = BaseDataChannelHandler.parse_struct_header(buf, header_cls)
        if header is None:
            return

        # Expect payload immediately following the header.
        size = header_cls.size()
        payload_start = size
        payload_end = payload_start + header.payload_len

        if payload_end != len(buf):
            LOGGER.warning(
                "Length mismatch in chunk (frame %d) on '%s': payload_len=%d, buf_len=%d",
                header.frame_id,
                self.name,
                header.payload_len,
                len(buf),
            )
            return

        payload = bytes(buf[payload_start:payload_end])

        # Per-chunk hook.
        self.on_chunk_received(header, payload)

        # Buffering logic.
        entry = self._buffers.get(header.frame_id)
        if entry is None:
            entry = self.FrameBuffer(
                total_chunks=header.total_chunks,
                chunks={},
                received=0,
            )
            self._buffers[header.frame_id] = entry

        if header.chunk_idx not in entry.chunks:
            entry.chunks[header.chunk_idx] = payload
            entry.received += 1

        if entry.received == entry.total_chunks:
            # Assemble and emit full frame.
            try:
                parts = [entry.chunks[i] for i in range(entry.total_chunks)]
            except KeyError:
                LOGGER.warning(
                    "Missing chunk(s) for frame %d on '%s' despite received=%d/%d",
                    header.frame_id,
                    self.name,
                    entry.received,
                    entry.total_chunks,
                )
                del self._buffers[header.frame_id]
                return

            del self._buffers[header.frame_id]
            frame_bytes: bytes = b"".join(parts)
            self.on_frame_received(header, frame_bytes)
