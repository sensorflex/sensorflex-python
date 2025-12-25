"""A library for supporting distributed computation."""

import struct
from typing import Any, Dict, List

import msgspec

from sensorflex import Node, Port


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

        self._encoder = msgspec.msgpack.Encoder()
        self._type_mapping = type_mapping

    def forward(self) -> None:
        # Network byte order (big-endian): 2s B H I  => magic, version, type_id, length
        format = "!2sBHI"
        # size = struct.calcsize(format)

        magic = b"SF"
        version = 1

        data = ~self.i_data
        tid = self._type_mapping[type(data)]
        payload = self._encoder.encode(data)
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
