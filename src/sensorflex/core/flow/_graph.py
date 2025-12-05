"""A library for building SensorFlex computation graph."""

from __future__ import annotations

import asyncio
import numpy as np
from threading import Thread
from numpy.typing import NDArray
from typing import Any, TypeVar, Tuple, List, Dict

from ._node import Node, Port


G = TypeVar("G", bound=Node)


class Graph:
    nodes: List[Node]
    edges: List[Tuple[Port, Port]]

    __batching: bool
    __node_edge_map: Dict[Node, List[Tuple[Port, Port]]]
    __ports_to_listen: set[Port]
    __ports_changed: List[Port]

    __event_queue: asyncio.Queue[Port]
    __loop: asyncio.AbstractEventLoop | None = None

    def __init__(self) -> None:
        self.nodes = []
        self.edges = []

        # Ports we care about
        self.__node_edge_map: Dict[Node, List[Tuple[Port, Port]]] = {}
        self.__ports_to_listen: set[Port] = set()

        # For batched (sync) updates
        self.__batching = False
        self.__ports_changed = []

        # For async/event-driven mode in a dedicated thread
        self.__event_queue = asyncio.Queue()
        self.__loop = None

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

    # class ListenerGraph(Graph):
    #     def __init__(self) -> None:
    #         super().__init__()

    def watch(self, port: Port) -> None:
        port.__graph_to_notify = self
        self.__ports_to_listen.add(port)

    def on_port_change(self, port: Port) -> None:
        """
        Called from wherever the port is updated. This might be in another thread.
        """
        if port not in self.__ports_to_listen:
            return

        if self.__batching:
            # Sync/batched mode: just collect.
            self.__ports_changed.append(port)
            return

        # IMPORTANT: this may be called from another thread
        loop = self.__loop
        if loop is not None and loop.is_running():
            # Schedule the queue put in a thread-safe way
            loop.call_soon_threadsafe(self.__event_queue.put_nowait, port)
        else:
            # Fallback if we somehow don't have a loop yet / same-thread use
            self.__event_queue.put_nowait(port)

    async def run_and_wait_forever(self) -> None:
        # Record which loop we are running on
        self.__loop = asyncio.get_running_loop()
        while True:
            _ = await self.__event_queue.get()
            self.run()

    def run_and_wait_forever_as_task(self) -> asyncio.Task:
        return asyncio.create_task(self.run_and_wait_forever())

    def update(self) -> ListenerGraphUpdateContext:
        return ListenerGraphUpdateContext(self)

    # def __ilshift__(self, edge: Tuple[Port, Port]) -> ListenerGraph:
    #     g = super().__ilshift__(edge)
    #     return cast(ListenerGraph, g)


class ListenerGraphUpdateContext:
    graph: Graph

    def __init__(self, graph: Graph):
        self.graph = graph
        self._prev_batching: bool | None = None

    def __enter__(self):
        # Save previous batching state so contexts can be nested safely.
        self._prev_batching = self.graph.__batching

        # Enter batching mode
        self.graph.__batching = True
        self.graph.__ports_changed.clear()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Leave batching mode
        self.graph.__batching = (
            self._prev_batching if self._prev_batching is not None else False
        )

        if self.graph.__ports_changed:
            # At least one watched port changed while in the context
            self.graph.run()
            self.graph.__ports_changed.clear()

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
