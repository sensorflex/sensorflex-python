from __future__ import annotations

from typing import Any, Callable
from pydantic import BaseModel
from sensorflex.core.flow._operator import Operator, switch


class X(Operator):
    class Input(BaseModel):
        x: int

    class Output(BaseModel):
        x: int
        y: int

    def run(self, v: Input) -> Output:
        return X.Output(x=v.x, y=v.x + 1)


class Y(Operator):
    class Input(BaseModel):
        x: int
        y: int

    class Output(BaseModel):
        z: float

    def run(self, v: Input) -> Output:
        return Y.Output(z=float(v.x + v.y))


class Add100(Operator):
    class Input(BaseModel):
        z: float

    class Output(BaseModel):
        z: float

    def run(self, v: Input) -> Output:
        return Add100.Output(z=v.z + 100.0)


class Sub100(Operator):
    class Input(BaseModel):
        z: float

    class Output(BaseModel):
        z: float

    def run(self, v: Input) -> Output:
        return Sub100.Output(z=v.z - 100.0)


def _cond_big(v: Y.Output) -> bool:
    return v.z > 10.0


def _debug_print(name: str) -> Callable[[Any], None]:
    def _inner(frame: Any) -> None:
        print(f"[{name}] {frame}")

    return _inner


if __name__ == "__main__":
    # Example pipeline:
    #   X
    #   | Y
    #   | Switch( cond ? Add100 : Sub100 )
    pipeline = (
        X()
        | Y()
        | switch(
            (_cond_big, Add100()),
            default=Sub100(),
        )
    )

    inp = X.Input(x=5)
    print("Input:", inp)

    out = pipeline(inp)
    print("Output pipeline:", out)

    # Example with tap + dict input
    pipeline2 = X().tap(_debug_print("after X")) | Y()
    out2 = pipeline2({"x": 3})  # dict gets validated into X.Input
    print("Output pipeline2:", out2)
