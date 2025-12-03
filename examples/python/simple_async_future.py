"""A simple example for using async Future objects."""

from asyncio import Future, create_task, sleep, run

from typing import Any

from sensorflex import Node, Graph, Port


class AsyncStepNode(Node):
    state: Port[str]
    f_fetch: Future | None = None

    def __init__(self, name: str | None = None) -> None:
        super().__init__(name)
        self.state = Port("None")

    def forward(self):
        if self.f_fetch is None:
            self.f_fetch = create_task(self.fetch_data())
            self.state <<= "Started"
        else:
            if self.f_fetch.done():
                self.state <<= "Done"
                self.f_fetch = None
            else:
                self.state <<= "Running"

    async def fetch_data(self):
        await sleep(5.0)


class PrintNode(Node):
    field: Port[Any]

    def __init__(self, name: str | None = None) -> None:
        super().__init__(name)
        self.field = Port(None)

    def forward(self):
        print(~self.field)


def get_graph():
    g = Graph()
    n1 = g << AsyncStepNode()
    n2 = g << PrintNode()
    g <<= n1.state >> n2.field

    return g


async def main():
    g = get_graph()

    for _ in range(8):
        g.run()
        await sleep(1)


if __name__ == "__main__":
    run(main())
