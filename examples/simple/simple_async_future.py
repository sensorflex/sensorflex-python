"""A simple example for using async Future objects."""

from asyncio import sleep, run
from typing import Any, Union

from sensorflex import Node, Graph, Port, FutureOp, FutureState


class AsyncStepNode(Node):
    state: Port[str]

    fetch_fop: FutureOp[str]

    def __init__(self, name: Union[str, None] = None) -> None:
        super().__init__(name)
        self.state = Port("None")
        self.fetch_fop = FutureOp(self.fetch_data)

    def forward(self):
        # For Python >= 3.10
        # match self.fetch_fop.step():
        #     case FutureState.STARTED:
        #         self.state <<= "Started"
        #     case FutureState.RUNNING:
        #         self.state <<= "Running"
        #     case FutureState.COMPLETED:
        #         res = self.fetch_fop.get_result()
        #         assert res is not None
        #         self.state <<= "Done" + res

        if self.fetch_fop is FutureState.STARTED:
            self.state <<= "Started"
        elif self.fetch_fop is FutureState.RUNNING:
            self.state <<= "Running"
        elif self.fetch_fop is FutureState.COMPLETED:
            res = self.fetch_fop.get_result()
            assert res is not None
            self.state <<= "Done" + res

    async def fetch_data(self) -> str:
        await sleep(5.0)
        return "123"


class PrintNode(Node):
    field: Port[Any]

    def __init__(self, name: Union[str, None] = None) -> None:
        super().__init__(name)
        self.field = Port(None)

    def forward(self):
        print(~self.field)


def get_graph():
    mp = (g := Graph()).main_pipeline
    mp += (n1 := AsyncStepNode())
    mp += (n2 := PrintNode())
    mp += n1.state >> n2.field

    return g


async def main():
    g = get_graph()

    for _ in range(8):
        g.run_main_pipeline()
        await sleep(1)


if __name__ == "__main__":
    run(main())
