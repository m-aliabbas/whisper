import json
import numpy as np
import sounddevice as sd
import timeit
from vosk import Model, KaldiRecognizer



class VoskTranscriptorAPI:
    def __init__(self, model_path='/home/ali/Desktop/idrak_work/whisper/WTranscriptor/model1'):
        self.model_path = model_path
        self.model = Model(self.model_path)
        self.sample_rate = 16000
        self.rec = KaldiRecognizer(self.model, self.sample_rate)
        self.silence_duration = 0
        self.total_duration = 0

    def generate_transcript_numpy(self, wave, pause_length=3, chunk_size=0.25, max_duration=10):
        int_data = np.int16(wave * 32768).tobytes()
        self.total_duration += chunk_size
        self.silence_duration += chunk_size

        if self.rec.AcceptWaveform(int_data):
            pass
        #     result = json.loads(self.rec.Result())
        #     if result.get('text'):
        #         self.silence_duration = 0
        #         # print(result["text"])
        #         return result['text'], True

        # Check for pause length
        if self.silence_duration >= pause_length:
            result = json.loads(self.rec.FinalResult())
            print("Pause detected.")
            self.rec = KaldiRecognizer(self.model, self.sample_rate)
            self.silence_duration = 0
            return result['text'], True

        # Check for total duration
        if self.total_duration >= max_duration:
            result = json.loads(self.rec.FinalResult())
            print("Max Time detected.")
            self.rec = KaldiRecognizer(self.model, self.sample_rate)
            self.total_duration = 0
            return result['text'], True

        return '', False