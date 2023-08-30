from fastapi.responses import HTMLResponse
import numpy as np
from WhisperASR import ASR
import sounddevice as sd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import numpy as np
import json

from WTranscriptor import WTranscriptor
config = {"sample_rate":16000,"duration_threshold":3,"vad_threshold":0.6,"model_path":"tiny.en"}
asr = ASR(config) 
app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            numpy_array = np.array(json.loads(data))
            transcript = asr.get_transcript(numpy_array)
            print(transcript)
            await websocket.send_text(transcript[1])
    except WebSocketDisconnect:
        print("WebSocket disconnected")


# app = FastAPI()

# # Initialize your WTranscriptor object
# # Make sure to import WTranscriptor and necessary configs before this line
# # transcriptor = WTranscriptor()

# # @app.websocket("/ws")
# # async def websocket_endpoint(websocket: WebSocket):
# #     await websocket.accept()
# #     try:
# #         while True:
# #             # Receiving audio data from the client
# #             audio_data = await websocket.receive_bytes()
            
# #             # Convert bytes to numpy array if needed (assuming 'int16' dtype for this example)
# #             audio_np_array = np.frombuffer(audio_data, dtype='int16')

# #             # Push data into WTranscriptor and check for the result
# #             result = transcriptor.push(audio_np_array, pause_type=1)
            
# #             # If the push method returns True, send the transcript back to the client
# #             if result:
# #                 transcript = transcriptor.transcript[1]  # Assuming the transcript is in the second position of the tuple
# #                 await websocket.send_text(f"Transcript: {transcript}")

# #                 # Clear the internal state of the transcriptor
# #                 transcriptor.refresh()

# #     except Exception as e:
# #         print(f"Error: {e}")
# #         await websocket.close()


# from fastapi import FastAPI, WebSocket
# from WTranscriptor import WTranscriptor
# import numpy as np


# app = FastAPI()
# transcriptor = WTranscriptor({"sample_rate": 16000, "duration_threshold": 3, "vad_threshold": 0.6, "model_path": "base.en"})

# @app.websocket("/audio/")
# async def audio_endpoint(websocket: WebSocket):
#     await websocket.accept()
#     while True:
#         raw_audio_block = await websocket.receive_bytes()
#         status = transcriptor.push(raw_audio_block)
#         if status:
#             transcript = transcriptor.transcript[1]
#             await websocket.send_text(f"True|{transcript}")
#             transcriptor.refresh()
#         else:
#             await websocket.send_text("False")




# from fastapi import FastAPI, WebSocket, WebSocketDisconnect

# app = FastAPI()

# class ConnectionManager:
#     def __init__(self):
#         self.active_connections = []

#     async def connect(self, websocket: WebSocket):
#         await websocket.accept()
#         self.active_connections.append(websocket)

#     def disconnect(self, websocket: WebSocket):
#         self.active_connections.remove(websocket)

# manager = ConnectionManager()
# config = {"sample_rate":16000, "duration_threshold":3, "vad_threshold":0.6, "model_path":"base.en"}
# transcriptor = WTranscriptor(config)

# @app.websocket("/audio/")
# async def audio_endpoint(websocket: WebSocket):
#     await manager.connect(websocket)
#     try:
        
#         while True:
#             audio_data = await websocket.receive_bytes()
#             # Replace with your actual WTranscriptor's push method
#             status = transcriptor.push(audio_data)

#             if status:
#                 transcript_data = transcriptor.transcript
#                 transcriptor.refresh()
#                 await websocket.send_text(f"True|{transcript_data}")
#             else:
#                 await websocket.send_text("False")

#     except WebSocketDisconnect:
#         manager.disconnect(websocket)
#         print("WebSocket disconnected.")

