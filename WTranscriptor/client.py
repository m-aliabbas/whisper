# import asyncio
# import json
# import wave
# import numpy as np
# import websockets

# async def send_audio_and_receive_transcript():
#     uri = "ws://localhost:8000/ws"
#     async with websockets.connect(uri) as websocket:

#         # Read wave file
#         wav_file = wave.open("audios/backy.wav", 'rb')
#         framerate = wav_file.getframerate()
#         n_frames = wav_file.getnframes()

#         # Number of frames to read for each 5-second chunk
#         frames_per_chunk = 10 * framerate

#         while wav_file.tell() < n_frames:
#             # Read audio data and convert to numpy array
#             audio_data = wav_file.readframes(min(frames_per_chunk, n_frames - wav_file.tell()))
#             audio_array = np.frombuffer(audio_data, dtype=np.int16)

#             # Serialize numpy array to JSON
#             serialized_array = json.dumps(audio_array.tolist())

#             # Send audio data to WebSocket server
#             await websocket.send(serialized_array)

#             # Receive and print transcript from WebSocket server
#             transcript = await websocket.recv()
#             print(f"Received transcript: {transcript}")

#         # Close wave file
#         wav_file.close()

# # Use the received transcript
# async def main():
#     await send_audio_and_receive_transcript()

# import asyncio
# import websockets
# import soundfile as sf
# from websockets.exceptions import ConnectionClosed

# async def audio_sender():
#     uri = "ws://110.93.240.107:8080/audio/"
#     file_path = "audios/backy.wav"
#     blocksize = 16000  # corresponds to 1 second

#     try:
#         async with websockets.connect(uri) as websocket:
#             with sf.SoundFile(file_path, 'r') as f:
#                 while True:
#                     data = f.buffer_read(blocksize, dtype='int16')
#                     if len(data) == 0:
#                         break
                    
#                     data_bytes = bytes(data)
#                     await websocket.send(data_bytes)

#                     status_message = await websocket.recv()
#                     if status_message.startswith("True"):
#                         _, transcript = status_message.split("|")
#                         print(f"Transcript received: {transcript}")

#     except ConnectionClosed:
#         print("WebSocket connection closed.")
#     except Exception as e:
#         print(f"An exception occurred: {e}")

# if __name__ == "__main__":
#     asyncio.get_event_loop().run_until_complete(audio_sender())
