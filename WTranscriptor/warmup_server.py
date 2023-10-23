import requests
import numpy as np
import soundfile as sf
import json

def send_audio_to_server(filename,server_address):
    # Read WAV file
    data, samplerate = sf.read(filename, dtype='int16')
    
    # Resample if the rate is not 16000
    if samplerate != 16000:
        # Resample the data to 16000
        from scipy.signal import resample_poly
        new_length = int(data.shape[0] * 16000 / samplerate)
        data = resample_poly(data, 16000, samplerate)
        samplerate = 16000

    # Convert data to list (which can be serialized to JSON)
    audio_data = data.tolist()

    # Prepare the payload
    payload = {
        "audio_data": json.dumps(audio_data)
    }

    # Send request to the server
    response = requests.post(f"http://{server_address}/asr_classify_ner", json=payload)

    # Parse response
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print(f"Error: {response.status_code}")
        return None

if __name__ == "__main__":
    filename = "audios/10sec.wav"
    for i in range(10):
        server_address = f'localhost:800{i}'
        print('Warming up' , server_address)
        result = send_audio_to_server(filename,server_address=server_address)
        if result:
            print(result)

