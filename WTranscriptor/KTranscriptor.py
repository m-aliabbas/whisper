
import json
import numpy as np
from vosk1 import VoskTranscriptorAPI


class KTranscriptor(object):
    def __init__(self, config=dict()):
        self.config = config
        print(self.config)

        # Instead of using a numpy array to concatenate audio data, use a list for efficiency.
        # It's faster to append to a list than to concatenate numpy arrays.
        self.audio_list = []
        
        self.asr = VoskTranscriptorAPI()
        self.transcript = None

    def push(self, raw_audio_block, pause_length=1, max_duration=10,pause_type='small',verbose=False):
        tmp_np = self.byte2np(raw_audio_block)
        self.audio_list.append(tmp_np)

        # Convert list of audio chunks to a numpy array for processing
        current_audio_data = np.hstack(self.audio_list)
        
        transcpt,status = self.asr.generate_transcript_numpy(wave=raw_audio_block, 
                                                              pause_length=3, 
                                                              chunk_size=0.25, 
                                                              max_duration=max_duration)

        if status:
            # If we get a transcription, clear the audio list to start fresh
            self.audio_list.clear()
            self.transcript = transcpt

        return status

    def byte2np(self, data):
        return np.frombuffer(data, dtype='int16')
    
    def refresh(self):
        pass