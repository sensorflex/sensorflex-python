"""A library for supporting distributed computation."""

from typing import Any
from sensorflex import Node, Port

from ._socket import WebSocketServerNode


class TransmitterNode(Node):
    i_data: Port[Any]
    o_data: Port[Any]

    def __init__(self, name: str | None = None) -> None:
        super().__init__(name)

    def forward(self) -> None:
        return super().forward()
