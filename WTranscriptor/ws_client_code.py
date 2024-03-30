import asyncio
import websockets

async def send_file(url, file_path):
    async with websockets.connect(url) as websocket:
        # Open the file in binary read mode
        with open(file_path, "rb") as file:
            data = file.read()
            await websocket.send(data)  # Send the entire file data as binary
        
        # Wait for a response from the server
        response = await websocket.recv()
        print(f"Server response: {response}")

# Replace 'localhost' and '8000' with your FastAPI server's host and port
url = "ws://localhost:8000/ws_file_transcribe1"
file_path = "audios/40sec.wav"  # Path of the file you want to send

# Run the client to send the file
asyncio.run(send_file(url, file_path))
