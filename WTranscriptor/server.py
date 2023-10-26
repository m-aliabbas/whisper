# Standard Library Imports
from typing import List
import json
import timeit
from classification_utils.path_config import *
import sys
sys.path.append(CLASSIFIER_MODULE_PATH)
# External Libraries
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import numpy as np
from WhisperASR import ASR
import gzip
from classification_utils.utils import *
# Initialize FastAPI app
app = FastAPI()


def compress_data(data):
    return gzip.compress(data)
suppress_low = [
    "Thank you",
    "Thanks for",
    "ike and ",
    "lease sub",
    "The end.",
    "ubscribe",
    "my channel",
    "the channel",
    "our channel",
    "ollow me on",
    "for watching",
    "hank you for watching",
    "for your viewing",
    "r viewing",
    "Amara",
    "next video",
    "full video",
    "ranslation by",
    "ranslated by",
    "ee you next week",
]
# Configuration for ASR
config = {
    "sample_rate": 16000,
    "duration_threshold": 3,
    "vad_threshold": 0.6,
    "model_path": "tiny.en",

}
asr = ASR(config)



class ConnectionManager:
    """Manages WebSocket connections."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        """Accepts and stores a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)

    async def disconnect(self, websocket: WebSocket):
        """Removes a WebSocket connection."""
        self.active_connections.remove(websocket)

    async def send_data(self, data: str, websocket: WebSocket):
        """Sends data through a given WebSocket connection."""
        await websocket.send_text(data)


manager = ConnectionManager()



# @app.websocket("/asr_classify")
# async def websocket_endpoint(websocket: WebSocket):
#     await manager.connect(websocket)
#     try:
#         while True:
#             data = await websocket.receive_text()
            
#             # Convert received data to numpy array
#             numpy_array = np.array(json.loads(data))
#             entity = None
#             # Get transcript
#             temp_transcript = [[], '']
#             classification_result = ''
#             transcript = asr.get_transcript(numpy_array)
#             if len(transcript[1]) < 5 and ('you' in transcript[1].lower()): 
#                 transcript = temp_transcript
#             elif len(transcript[1]) < 12 and ('thank you' in transcript[1].lower()):
#                 transcript = temp_transcript
#             else:
#                 transcript = transcript
#                 classification_result = get_classification(transcript[1])
#                 entity = get_entity(transcript[1])
            
#             # print(classification_result)
#             # Send back a structured response
#             response_data = {
#                 "status": "success",
#                 "transcript": {'transcript':transcript[1],'intent':classification_result,
#                                'entity':entity}
#             }
#             await manager.send_data(json.dumps(response_data), websocket)

#     except WebSocketDisconnect:
#         manager.disconnect(websocket)
#         print("WebSocket disconnected")
#     except Exception as e:
#         print(f"Error occurred: {e}")

@app.websocket("/asr_classify_chunks")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    audio_buffer = bytearray()  # Use bytearray to accumulate binary data

    try:
        while True:
            # print(len(audio_buffer))
            data = await websocket.receive()
            # print(data)
            # Check for the type of message received
            if 'text' in data:
                text_data = data['text']
                # print(text_data)
                # If the completion signal is received, convert buffer to numpy array
                if text_data == "E":
                    # print(len(audio_buffer))
                    numpy_array = np.frombuffer(audio_buffer, dtype=np.int16)  # Assuming float32 dtype for audio
                    # print(numpy_array)
                    # Clear the audio buffer for the next stream
                    audio_buffer = bytearray()
                    hal_flag = False
                    entity = None
                    # Get transcript
                    temp_transcript = [[], '']
                    classification_result = ''
                    transcript = asr.get_transcript(numpy_array)
                    for hal in suppress_low:
                        if hal in transcript[1]:
                            hal_flag = True
                    if hal_flag:
                        transcript = temp_transcript
                        hal_flag = False
                    elif len(transcript[1]) < 5 and ('you' in transcript[1].lower()): 
                        transcript = temp_transcript
                    elif len(transcript[1]) < 12 and ('thank you' in transcript[1].lower()):
                        transcript = temp_transcript
                    else:
                        transcript = transcript
                        classification_result = get_classification(transcript[1])
                        entity = get_entity(transcript[1])

                    # Send back a structured response
                    response_data = {
                        "status": "success",
                        "transcript": {
                            'transcript': transcript[1],
                            'intent': classification_result,
                            'entity': entity
                        }
                    }
                    compressed_data = compress_data(json.dumps(response_data).encode('utf-8'))
                    await manager.send_data(compressed_data, websocket)

                elif text_data == "P":
                    # Handle other text data if needed
                    response_data = {
                        "status": "P",
                    }
                    compressed_data = compress_data(json.dumps(response_data).encode('utf-8'))
                    await manager.send_data(compressed_data, websocket)
                
                else:
                    break
                    

            elif 'bytes' in data:
                # Append binary data to audio buffer
                audio_buffer.extend(data['bytes'])
                # print(audio_buffer)

            else:
                break
            


    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print("WebSocket disconnected")
        
    except Exception as e:
        print(f"Error occurred: {e}")



