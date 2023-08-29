from fastapi.responses import HTMLResponse
import numpy as np
from WhisperASR import ASR
import sounddevice as sd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import numpy as np
import json


config = {"sample_rate":16000,"duration_threshold":3,"vad_threshold":0.6,"model_path":"base.en"}
asr = ASR(config) 
app = FastAPI()
print('Yes')

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