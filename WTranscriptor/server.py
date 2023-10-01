# Standard Library Imports
from typing import List
import json
from classification_utils.path_config import *
import sys
sys.path.append(CLASSIFIER_MODULE_PATH)
# External Libraries
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import numpy as np
from WhisperASR import ASR
from classification_utils.utils import *
# Initialize FastAPI app
app = FastAPI()

# Configuration for ASR
config = {
    "sample_rate": 16000,
    "duration_threshold": 3,
    "vad_threshold": 0.6,
    "model_path": "base.en"
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

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            
            # Convert received data to numpy array
            numpy_array = np.array(json.loads(data))
            
            # Get transcript
            transcript = asr.get_transcript(numpy_array)
            
            # Send back a structured response
            response_data = {
                "status": "success",
                "transcript": transcript[1]
            }
            await manager.send_data(json.dumps(response_data), websocket)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print("WebSocket disconnected")
    except Exception as e:
        print(f"Error occurred: {e}")

@app.websocket("/ws_classify")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            
            # Convert received data to numpy array
            numpy_array = np.array(json.loads(data))
            
            # Get transcript
            temp_transcript = [[], '']
            classification_result = ''
            transcript = asr.get_transcript(numpy_array)
            if len(transcript[1]) < 5 and ('you' in transcript[1].lower()): 
                transcript = temp_transcript
            elif len(transcript[1]) < 12 and ('thank you' in transcript[1].lower()):
                transcript = temp_transcript
            else:
                transcript = transcript
                classification_result = get_classification(transcript[1])
                
            
            print(classification_result)
            # Send back a structured response
            response_data = {
                "status": "success",
                "transcript": {'transcript':transcript[1],'intent':classification_result}
            }
            await manager.send_data(json.dumps(response_data), websocket)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print("WebSocket disconnected")
    except Exception as e:
        print(f"Error occurred: {e}")

