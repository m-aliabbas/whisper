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

# @app.websocket("/ws_classify")
# async def websocket_endpoint(websocket: WebSocket):
#     await manager.connect(websocket)
#     try:
#         while True:
#             data = await websocket.receive_text()
            
#             # Convert received data to numpy array
#             numpy_array = np.array(json.loads(data))
            
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
                
            
#             print(classification_result)
#             # Send back a structured response
#             response_data = {
#                 "status": "success",
#                 "transcript": {'transcript':transcript[1],'intent':classification_result}
#             }
#             await manager.send_data(json.dumps(response_data), websocket)

#     except WebSocketDisconnect:
#         manager.disconnect(websocket)
#         print("WebSocket disconnected")
#     except Exception as e:
#         print(f"Error occurred: {e}")


@app.websocket("/asr_classify")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            
            # Convert received data to numpy array
            numpy_array = np.array(json.loads(data))
            entity = None
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
                entity = get_entity(transcript[1])
            
            print(classification_result)
            # Send back a structured response
            response_data = {
                "status": "success",
                "transcript": {'transcript':transcript[1],'intent':classification_result,
                               'entity':entity}
            }
            await manager.send_data(json.dumps(response_data), websocket)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print("WebSocket disconnected")
    # except Exception as e:
    #     print(f"Error occurred: {e}")


@app.post("/asr_process")
async def process_asr(request: Request):
    try:
        # Receive JSON data
        data = await request.json()
        # print(data)
        # Check if 'audio_data' key is in the received data
        if "audio_data" not in data:
            raise HTTPException(status_code=400, detail="audio_data key not found in request body")
        # numpy_array = np.array(json.loads(data["audio_data"]))
        # print(numpy_array)
        # Convert received data to numpy array
        numpy_array = np.array(data["audio_data"])

        # Get transcript
        transcript = asr.get_transcript(numpy_array)
        
        # Return a structured response
        response_data = {
            "status": "success",
            "transcript": transcript[1]
        }
        return JSONResponse(content=response_data)
    except Exception as e:
        return HTTPException(status_code=500, detail=f"Error occurred: {e}")
    
@app.post("/asr_classify")
async def classify_asr(request: Request):
    t1 = timeit.timeit()
    # Receive JSON data
    data = await request.json()
    # print(data)
    # Check if 'audio_data' key is in the received data
    if "audio_data" not in data:
        raise HTTPException(status_code=400, detail="audio_data key not found in request body")

    # Convert received data to numpy array
    numpy_array = np.array(json.loads(data["audio_data"]))

    # Get transcript
    temp_transcript = [[], '']
    classification_result = ''
    transcript = asr.get_transcript(numpy_array)
    # print(transcript)
    if len(transcript[1]) < 5 and ('you' in transcript[1].lower()): 
        transcript = temp_transcript
    elif len(transcript[1]) < 12 and ('thank you' in transcript[1].lower()):
        transcript = temp_transcript
    else:
        transcript = transcript
        classification_result = get_classification(transcript[1])
    t2 = timeit.timeit()
    print('Time taken', t2-t1)
    # Return a structured response
    response_data = {
        "status": "success",
        "transcript": {'transcript':transcript[1], 'intent':classification_result}
    }
    return JSONResponse(content=response_data)


@app.post("/asr_classify_ner")
async def classify_asr(request: Request):
    t1 = timeit.timeit()
    # Receive JSON data
    data = await request.json()
    # print(data)
    # Check if 'audio_data' key is in the received data
    if "audio_data" not in data:
        raise HTTPException(status_code=400, detail="audio_data key not found in request body")

    # Convert received data to numpy array
    numpy_array = np.array(json.loads(data["audio_data"]))

    # Get transcript
    temp_transcript = [[], '']
    classification_result = ''
    transcript = asr.get_transcript(numpy_array)
    # print(transcript)
    if len(transcript[1]) < 5 and ('you' in transcript[1].lower()): 
        transcript = temp_transcript
    elif len(transcript[1]) < 12 and ('thank you' in transcript[1].lower()):
        transcript = temp_transcript
    else:
        transcript = transcript
        classification_result = get_classification(transcript[1])
        entity = get_entity(transcript[1])
        print('Entity',entity)
    t2 = timeit.timeit()
    print('Time taken', t2-t1)
    # Return a structured response
    response_data = {
        "status": "success",
        "transcript": {'transcript':transcript[1], 'intent':classification_result,
                       'entity':entity}
    }
    return JSONResponse(content=response_data)
