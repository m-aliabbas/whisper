import asyncio
import json
import wave
import numpy as np
import websockets

async def send_audio_and_receive_transcript():
    uri = "ws://localhost:8000/ws"
    async with websockets.connect(uri) as websocket:

        # Read wave file
        wav_file = wave.open("audios/backy.wav", 'rb')
        framerate = wav_file.getframerate()
        n_frames = wav_file.getnframes()

        # Number of frames to read for each 5-second chunk
        frames_per_chunk = 10 * framerate

        while wav_file.tell() < n_frames:
            # Read audio data and convert to numpy array
            audio_data = wav_file.readframes(min(frames_per_chunk, n_frames - wav_file.tell()))
            audio_array = np.frombuffer(audio_data, dtype=np.int16)

            # Serialize numpy array to JSON
            serialized_array = json.dumps(audio_array.tolist())

            # Send audio data to WebSocket server
            await websocket.send(serialized_array)

            # Receive and print transcript from WebSocket server
            transcript = await websocket.recv()
            print(f"Received transcript: {transcript}")

        # Close wave file
        wav_file.close()

# Use the received transcript
async def main():
    await send_audio_and_receive_transcript()

# Run the event loop
if __name__ == '__main__':
    asyncio.run(main())
