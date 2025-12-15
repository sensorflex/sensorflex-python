#!/usr/bin/env python3
"""
Combined WebSocket test - runs server and client together with multiple sends
"""

import asyncio
import websockets
import numpy as np
import time


# Server handler
async def handle_client(websocket):
    print("[Server] Client connected")

    receive_times = []

    try:
        while True:
            receive_start = time.perf_counter()
            data = await websocket.recv()
            receive_end = time.perf_counter()

            receive_time = (receive_end - receive_start) * 1000
            receive_times.append(receive_time)
            data_size_mb = len(data) / (1024 * 1024)

            print(f"[Server] Received {data_size_mb:.2f} MB in {receive_time:.2f} ms")

            await websocket.send(f"OK")
    except websockets.exceptions.ConnectionClosed:
        if receive_times:
            print(f"\n[Server] Statistics ({len(receive_times)} receives):")
            print(f"  Min: {min(receive_times):.2f} ms")
            print(f"  Max: {max(receive_times):.2f} ms")
            print(f"  Avg: {sum(receive_times) / len(receive_times):.2f} ms")
        print("[Server] Client disconnected")


# Server task
async def run_server():
    async with websockets.serve(
        handle_client, "localhost", 8765, max_size=10 * 1024 * 1024, compression=None
    ):
        print("[Server] Started on ws://localhost:8765")
        await asyncio.Future()  # Run forever


# Client task
async def run_client(num_sends=10):
    # Wait for server to start
    await asyncio.sleep(0.5)

    print(f"[Client] Creating 1920x1080x3 array...")
    data_array = np.random.randint(0, 256, size=(1080, 1920, 3), dtype=np.uint8)
    data_bytes = data_array.tobytes()
    data_size_mb = len(data_bytes) / (1024 * 1024)

    print(f"[Client] Array size: {data_size_mb:.2f} MB ({len(data_bytes):,} bytes)")
    print(f"[Client] Will send {num_sends} times")
    print(f"[Client] Connecting...")

    async with websockets.connect(
        "ws://localhost:8765", max_size=10 * 1024 * 1024, compression=None
    ) as websocket:
        print(f"[Client] Connected! Starting sends...\n")

        send_times = []

        for i in range(num_sends):
            send_start = time.perf_counter()
            await websocket.send(data_bytes)
            send_end = time.perf_counter()

            send_time = (send_end - send_start) * 1000
            send_times.append(send_time)

            print(
                f"[Client] Send #{i + 1}: {send_time:.2f} ms ({data_size_mb / (send_time / 1000):.2f} MB/s)"
            )

            await websocket.recv()  # Wait for ack

        print(f"\n{'=' * 60}")
        print(f"[Client] Statistics ({num_sends} sends):")
        print(f"  Min send time: {min(send_times):.2f} ms")
        print(f"  Max send time: {max(send_times):.2f} ms")
        print(f"  Avg send time: {sum(send_times) / len(send_times):.2f} ms")
        print(
            f"  Avg throughput: {data_size_mb / (sum(send_times) / len(send_times) / 1000):.2f} MB/s"
        )
        print(f"{'=' * 60}\n")


async def main(num_sends=100):
    # Run server and client concurrently
    server_task = asyncio.create_task(run_server())
    client_task = asyncio.create_task(run_client(num_sends))

    # Wait for client to finish
    await client_task

    # Cancel server
    server_task.cancel()
    try:
        await server_task
    except asyncio.CancelledError:
        pass

    print("\nTest completed!")


if __name__ == "__main__":
    import sys

    # Get number of sends from command line argument, default to 10
    num_sends = int(sys.argv[1]) if len(sys.argv) > 1 else 10

    print(f"Running WebSocket performance test with {num_sends} sends\n")
    asyncio.run(main(num_sends))
