import asyncio
import websockets
import os

async def send_file_and_receive_transcription(uri, file_path):
    # Open the WebSocket connection
    async with websockets.connect(uri) as websocket:
        # Read the .wav file's binary data
        with open(file_path, 'rb') as file:
            binary_data = file.read()
            # Send the binary data over the WebSocket
            # await websocket.send(binary_data)
            await websocket.send(binary_data)
            print("[+] File sent, closing connection...")
            await websocket.close()

            print("[+] File sent, waiting for transcription...")
        
        # Receive the transcription result from the server
        transcription = await websocket.recv()
        print("[+] Transcription received:")
        print(transcription)

# The URI of the WebSocket endpoint
uri = "ws://localhost:8000/ws_file_transcribe"

# Path to the .wav file you want to send
file_path = "audios/10sec.wav"

# Running the asyncio event loop
asyncio.run(send_file_and_receive_transcription(uri, file_path))
