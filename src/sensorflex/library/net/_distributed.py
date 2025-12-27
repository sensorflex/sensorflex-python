"""A library for supporting distributed computation."""

from __future__ import annotations

import struct
from typing import Any, Callable, Dict, List, Type, cast

import msgspec
from websockets.typing import Data

from sensorflex import Node, Port


def get_msgpack_coder_transforms(
    *data_models: Type[Any],
) -> tuple[
    Callable[[Any], bytes],
    Callable[[Data], Any],
]:
    """
    Create encoder/decoder functions for the given data models.

    Args:
        *data_models: Variable number of data model types to support

    Returns:
        A tuple of (encode function, decode function)
    """
    encoder_mapping: Dict[Type[Any], int] = {}
    decoder_mapping: Dict[int, Type[Any]] = {}

    for i, v in enumerate(data_models):
        encoder_mapping[v] = i
        decoder_mapping[i] = v

    def encode(data: Any) -> bytes:
        return sf_encode(data, encoder_mapping)

    def decode(payload: Data) -> Any:
        payload_bytes = cast(bytes, payload)
        return sf_decode(payload_bytes, decoder_mapping)

    return encode, decode


def sf_encode(data: Any, type_mapping: Dict[Any, Any]) -> bytes:
    format = "!2sBHI"
    # size = struct.calcsize(format)

    magic = b"SF"
    version = 1

    tid = type_mapping[type(data)]
    payload = msgspec.msgpack.encode(data)
    header = struct.pack(format, magic, version, tid, len(payload))

    return header + payload


def sf_decode(data: bytes, type_mapping: Dict[Any, Any]) -> Any:
    format = "!2sBHI"
    size = struct.calcsize(format)

    magic = b"SF"
    version = 1

    buf = data

    if len(buf) < size:
        raise ValueError("need more bytes for header")

    magic, ver, tid, length = struct.unpack(format, buf[:size])
    if magic != magic:
        raise ValueError(f"bad magic: {magic!r}")
    if ver != version:
        raise ValueError(f"unsupported version: {ver}")
    msg_type = type_mapping.get(tid)
    if msg_type is None:
        raise ValueError(f"unknown type id: {tid}")

    total = size + length
    if len(buf) < total:
        raise ValueError("need more bytes for payload")

    payload = buf[size:total]
    obj = msgspec.msgpack.decode(payload, type=msg_type)

    return obj


def get_msgpack_encoder_decoder_nodes(data_models: List[Any]):
    encoder_mapping, decoder_mapping = {}, {}

    for i, v in enumerate(data_models):
        encoder_mapping[v] = i
        decoder_mapping[i] = v

    encoder = MessagePackEncoder(encoder_mapping)
    decoder = MessagePackDecoder(decoder_mapping)
    return encoder, decoder


class MessagePackEncoder(Node):
    i_data: Port[Any]
    o_bytes: Port[bytes]

    def __init__(self, type_mapping: Dict[Any, Any], name: str | None = None) -> None:
        super().__init__(name)
        self.i_data = Port(None)
        self.o_bytes = Port(None)

        self._type_mapping = type_mapping

    def forward(self) -> None:
        # Network byte order (big-endian): 2s B H I  => magic, version, type_id, length
        format = "!2sBHI"
        # size = struct.calcsize(format)

        magic = b"SF"
        version = 1

        data = ~self.i_data
        tid = self._type_mapping[type(data)]
        payload = msgspec.msgpack.encode(data)
        header = struct.pack(format, magic, version, tid, len(payload))

        self.o_bytes <<= header + payload


class MessagePackDecoder(Node):
    i_bytes: Port[bytes]
    o_data: Port[Any]

    def __init__(self, type_mapping: Dict[Any, Any], name: str | None = None) -> None:
        super().__init__(name)
        self.i_bytes = Port(None)
        self.o_data = Port(None)

        self._encoder = msgspec.msgpack.Decoder()
        self._type_mapping = type_mapping

    def forward(self) -> None:
        # Network byte order (big-endian): 2s B H I  => magic, version, type_id, length
        format = "!2sBHI"
        size = struct.calcsize(format)

        magic = b"SF"
        version = 1

        buf = ~self.i_bytes

        if len(buf) < size:
            raise ValueError("need more bytes for header")

        magic, ver, tid, length = struct.unpack(format, buf[:size])
        if magic != magic:
            raise ValueError(f"bad magic: {magic!r}")
        if ver != version:
            raise ValueError(f"unsupported version: {ver}")
        msg_type = self._type_mapping.get(tid)
        if msg_type is None:
            raise ValueError(f"unknown type id: {tid}")

        total = size + length
        if len(buf) < total:
            raise ValueError("need more bytes for payload")

        payload = buf[size:total]
        obj = msgspec.msgpack.decode(payload, type=msg_type)

        self.o_data <<= obj
