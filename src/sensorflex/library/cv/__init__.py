"""CV library nodes."""

from ._webcam import (
    ImageCodec,
    ImageDecodeNode,
    ImageEncodeNode,
    RandImgNode,
    WebcamNode,
)

__all__ = [
    "WebcamNode",
    "RandImgNode",
    "ImageCodec",
    "ImageEncodeNode",
    "ImageDecodeNode",
]
