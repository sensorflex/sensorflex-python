"""A library for operators"""

from __future__ import annotations

from dataclasses import dataclass
from numpy.typing import NDArray
from typing import Any


class Operator:
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Must implement forward.")


@dataclass
class BgrToRgbOp(Operator):
    bgr: NDArray
    use_resizing: bool = False

    def forward(self) -> NDArray:
        rgb = self.bgr[::, [2, 1, 0]]
        return rgb
