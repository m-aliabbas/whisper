# import json
# import numpy as np
# import soundfile as sf
# from vosk import Model, KaldiRecognizer

# # Parameters
# PAUSE_LENGTH = 2  # seconds

# model = Model("model1/")

# # Set the sample rate and chunk size
# SAMPLE_RATE = 16000
# CHUNK = 4000  # 0.25 seconds per chunk
# FILENAME = "audios/backy.wav"

# rec = KaldiRecognizer(model, SAMPLE_RATE)
# silence_duration = 0
# total_duration = 0

# with sf.SoundFile(FILENAME, mode='r') as file:
#     for audio_chunk in file.blocks(blocksize=CHUNK, fill_value=0):
#         int_data = np.int16(audio_chunk * 32768).tobytes()

#         if rec.AcceptWaveform(int_data):
#             result = json.loads(rec.Result())
#             if result.get('text'):
#                 print(result['text'])
#                 silence_duration = 0
#             else:
#                 silence_duration += CHUNK / SAMPLE_RATE

#             # Check for pause length
#             if silence_duration >= PAUSE_LENGTH:
#                 print("Pause detected. Transcript so far:", rec.FinalResult())
#                 rec = KaldiRecognizer(model, SAMPLE_RATE)
#                 silence_duration = 0

#             total_duration += CHUNK / SAMPLE_RATE

#             # Check for total duration of 10 seconds
#             if total_duration >= 10:
#                 print("10-second interval. Transcript so far:", rec.FinalResult())
#                 rec = KaldiRecognizer(model, SAMPLE_RATE)
#                 total_duration = 0

# print("Final transcript:", rec.FinalResult())

import json
import numpy as np
import sounddevice as sd
from vosk import Model, KaldiRecognizer

# Parameters
PAUSE_LENGTH = 2  # seconds

model = Model("model1/")

# Set the sample rate and chunk size
SAMPLE_RATE = 16000
CHUNK = 4000  # 0.25 seconds per chunk

rec = KaldiRecognizer(model, SAMPLE_RATE)
silence_duration = 0
total_duration = 0

with sd.InputStream(samplerate=SAMPLE_RATE, channels=1) as stream:
    print("Recording... Press Ctrl+C to stop.")

    try:
        while True:
            audio_chunk, overflowed = stream.read(CHUNK)
            int_data = np.int16(audio_chunk * 32768).tobytes()

            # Increment the total_duration and silence_duration irrespective of recognition result
            total_duration += CHUNK / SAMPLE_RATE
            silence_duration += CHUNK / SAMPLE_RATE

            if rec.AcceptWaveform(int_data):
                result = json.loads(rec.Result())
                if result.get('text'):
                    # print(result['text'])
                    silence_duration = 0

            # Check for pause length
            if silence_duration >= PAUSE_LENGTH:
                result = json.loads(rec.Result())
                print("Pause detected. Transcript so far:")
                rec = KaldiRecognizer(model, SAMPLE_RATE)
                silence_duration = 0

            # Check for total duration of 10 seconds
            # print("Total duration", total_duration)   
            if total_duration >= 10:
                result = json.loads(rec.Result())
                print("Max Time detected. Transcript so far:")
                rec = KaldiRecognizer(model, SAMPLE_RATE)
                total_duration = 0


    except KeyboardInterrupt:
        pass

print("Final transcript:", rec.FinalResult())
