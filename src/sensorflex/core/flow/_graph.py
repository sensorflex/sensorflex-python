"""A library for building SensorFlex computation graph."""

from __future__ import annotations

import asyncio
import numpy as np
from threading import Thread
from numpy.typing import NDArray
from typing import Any, TypeVar, Tuple, List, Dict, cast

from ._node import Node, Port


G = TypeVar("G", bound=Node)


class Graph:
    def __init__(self) -> None:
        self.nodes: List[Node] = []
        self.edges = []

        self.__node_edge_map: Dict[Node, List[Tuple[Port, Port]]] = {}

    def add(self, node: G) -> G:
        node.__register_ports__()
        self.nodes.append(node)
        return node

    def connect(self, left_node, right_node) -> None:
        pass

    def run(self):
        # async with asyncio.TaskGroup() as tg:
        for node in self.nodes:
            if node in self.__node_edge_map:
                if self.__node_edge_map[node] is not None:
                    for port_a, port_b in self.__node_edge_map[node]:
                        # Flow data between ports
                        port_b.value = port_a.value

            # if inspect.iscoroutinefunction(node.forward):
            #     tg.create_task(node.forward())
            # else:
            node.forward()

    def run_in_thread(self) -> Thread:
        t = Thread(target=self.run)
        t.start()
        return t

    def __lshift__(self, node: G) -> G:
        self.add(node)
        return node

    def __ilshift__(self, edge: Tuple[Port, Port]) -> Graph:
        # Process edges
        self.edges.append(edge)

        receiver_node = edge[1].parent_node
        if receiver_node not in self.__node_edge_map:
            self.__node_edge_map[receiver_node] = [edge]
        else:
            self.__node_edge_map[receiver_node].append(edge)

        return self

    def __enter__(self) -> Graph:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class ListenerGraph(Graph):
    def __init__(self) -> None:
        super().__init__()

        # Ports we care about
        self.ports_to_listen: set[Port] = set()

        # For batched (sync) updates
        self._batching: bool = False
        self.ports_changed: list[Port] = []

        # For async/event-driven mode in a dedicated thread
        self._event_queue: asyncio.Queue[Port] = asyncio.Queue()

        # The asyncio event loop that will run run_and_wait_forever()
        self._loop: asyncio.AbstractEventLoop | None = None

    def watch(self, port: Port) -> None:
        port.graph_to_notify = self
        self.ports_to_listen.add(port)

    def on_port_change(self, port: Port) -> None:
        """
        Called from wherever the port is updated. This might be in another thread.
        """
        if port not in self.ports_to_listen:
            return

        if self._batching:
            # Sync/batched mode: just collect.
            self.ports_changed.append(port)
            return

        # IMPORTANT: this may be called from another thread
        loop = self._loop
        if loop is not None and loop.is_running():
            # Schedule the queue put in a thread-safe way
            loop.call_soon_threadsafe(self._event_queue.put_nowait, port)
        else:
            # Fallback if we somehow don't have a loop yet / same-thread use
            self._event_queue.put_nowait(port)

    async def run_and_wait_forever(self) -> None:
        # Record which loop we are running on
        self._loop = asyncio.get_running_loop()
        while True:
            _ = await self._event_queue.get()
            self.run()

    def run_and_wait_forever_as_task(self) -> asyncio.Task:
        return asyncio.create_task(self.run_and_wait_forever())

    def update(self) -> ListenerGraphUpdateContext:
        return ListenerGraphUpdateContext(self)

    def __ilshift__(self, edge: Tuple[Port, Port]) -> ListenerGraph:
        g = super().__ilshift__(edge)
        return cast(ListenerGraph, g)


class ListenerGraphUpdateContext:
    graph: ListenerGraph

    def __init__(self, graph: ListenerGraph):
        self.graph = graph
        self._prev_batching: bool | None = None

    def __enter__(self):
        # Save previous batching state so contexts can be nested safely.
        self._prev_batching = self.graph._batching

        # Enter batching mode
        self.graph._batching = True
        self.graph.ports_changed.clear()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Leave batching mode
        self.graph._batching = (
            self._prev_batching if self._prev_batching is not None else False
        )

        if self.graph.ports_changed:
            # At least one watched port changed while in the context
            self.graph.run()
            self.graph.ports_changed.clear()

        # Do not suppress exceptions
        return False


if __name__ == "__main__":
    # Define your nodes
    class ImageLoadingNode(Node):
        # path: Port[str] = Port(None)
        i: Port[int] = Port(0)
        bgr: Port[NDArray] = Port(None)

        def forward(self):
            # do something with ~self.path
            x = np.array([3, 2, 1], dtype=np.uint8) + ~self.i
            x = x.reshape((1, 1, 3))
            self.bgr <<= x

    class ImageTransformationNode(Node):
        bgr: Port[NDArray] = Port(None)
        rgb: Port[NDArray] = Port(None)

        def forward(self):
            bgr = ~self.bgr
            rgb = bgr[:, :, [2, 1, 0]]
            self.rgb <<= rgb

    class PrintNode(Node):
        field: Port[Any] = Port(None)

        def forward(self):
            print(~self.field)

    # Define a graph
    g = Graph()
    n1 = g << ImageLoadingNode()
    n2 = g << ImageTransformationNode()
    g <<= n1.bgr >> n2.bgr

    n3 = g << PrintNode()
    g <<= n3.field << n2.bgr

    # Now execute
    for i in range(5):
        n1.i <<= i
        g.run()
