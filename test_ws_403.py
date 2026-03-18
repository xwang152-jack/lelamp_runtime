import asyncio
import websockets

async def main():
    try:
        async with websockets.connect("ws://localhost:8000/api/ws/lelamp") as ws:
            print("Connected to /api/ws/lelamp")
            msg = await ws.recv()
            print("Received:", msg)
    except Exception as e:
        print(f"Error: {e}")

asyncio.run(main())
