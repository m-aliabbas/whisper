import torch
import numpy as np
import soundfile as sf
from WhisperASR import ASR
import enums
import sounddevice as sd
import math
from VAD_Interface import VAD_Interface

class WTranscriptor(object):
    """
    Can take streaming raw audio blocks and convert into transcript

    This module can take the data from the sounddevice rawstreams as input. 
    It accumulates the raw audio blocks and converts them into transcripts.
    The accumulation is done based on the following conditions:
    
    1. The module keeps accumulating data, using the push function, till it reaches a duration threshold (e.g 3 secs)
    2. If a pause is detected (silence for a few seconds) the push function returns true. 
    3. If data is accumulated to more than maximum allowable duration, the push function returns true
    4. The user is expected to store the transcript at each true return of push and call the refresh function to free up memory.  
    
    
    Silence Detection/Pause Detection:
    ;For Pause Detection we are gonna use silero_vad (https://pytorch.org/hub/snakers4_silero-vad_vad/)
    ;vad_threshold is 0.6 which is obtain using different trail and erros i.e experiments
    ;After 3 seconds we are checking for speech. And Comparing the last speech end with current sample stamps
    ;If there is gap of 3 seconds we call it 3 sec pause
    
    """
    def __init__(self, config=dict()) -> None:
        """
        The constructor requires a config dictionary to set the configs. 

        All configs have a default value in case the config value is not found in the dictionary.
        A sample config is part of this module
        """
        config = {"sample_rate":16000,"duration_threshold":3,"vad_threshold":0.6}
        self.config = config
        self.asr = ASR(config) 
        self.vad = VAD_Interface(config=config)
        self.max_allowable_duration = config.get("maximum_allowable_duration", 10) # max duration of input audio to transcribe in seconds
        self.samplerate = config.get("samplerate", 16000.0)
        self.data_array = np.array([])
        self.cuda_device = config.get("cuda_device", "cpu")
        self.duration_threshold = config.get("duration_threshold", 3)  # after this many seconds, pass the data through the model
        self.duration_threshold_delta = config.get("duration_threshold_delta", 1) # increase in duration thresold, for next iteration. 
        if not "enum" in config:
            config["enum"] = dict()
        
        self.status = False
        self.transcript = None
        self.amplitude = np.iinfo(np.int16).max
        
        # self.warmup()

    def push(self, raw_audio_block, pause_type=1, is_greedy=False, verbose=False,last_block=False):
        """
        The main function of the Transcriptor module. For an example usage, see the code in __main__ in Transcriptor.py

        Args:
            raw_audio_block: Output from sounddevice raw stream of arbitrary blocksizey.
            
        """
        gen_transcript = False
        #obtaining audio from bytes
        tmp_np = self.byte2np(raw_audio_block)
        pause_status = False
        self.data_array = np.hstack((self.data_array,tmp_np))
        duration  = len(self.data_array) / self.samplerate #duration 16000/16000=1s\
        speech_dict=None
        if duration >= self.duration_threshold: #if duration is larger than 3s
            data = self.data_array
            #passing data from VAD Model
            if len(data) % int(self.samplerate) == 0: #running only when a sec ticks
                pause_status = self.vad.pause_status(data=self.data_array)
            if pause_status: #if speech detected
                print('[+] Pause Detected')
                self.status=True
                gen_transcript = True

            if duration > self.max_allowable_duration: #10 second acheived
                print("[-] Max Limit Exceed")
                self.status = True
                gen_transcript = True
        
        if gen_transcript or last_block: #if last block of file or condition of pause or max duration meet
            transcript = self.asr.get_transcript(self.data_array)
            self.transcript = transcript
            self.status=True
            
        return self.status

    def refresh(self):
        self.data_array = np.array([])
        self.transcript = None
        self.status = False
        self.duration_threshold = self.config.get("duration_threshold", 3)

    def byte2np(self, data):
        return np.frombuffer(data, dtype='int16')


# -- For testing the module independantly
if __name__ == "__main__":

    filepath  = "/home/ali/Desktop/idrak_work/transcriptor_module-transcriptor-module/WTranscriptor/audios/gettysburg.wav"
    file_object =  sf.SoundFile(filepath)
    blocksize = 8000
    dtype = 'int16'
    samplerate = file_object.samplerate
    last_block_index = int(len(file_object)/blocksize)
    config=dict()
    transcriptor = WTranscriptor()
    transcpt=''
    raw_data = file_object.buffer_read(blocksize, dtype=dtype)
    block_index = 1
    is_last_processing_block=False
    import timeit

    # start = timeit.default_timer()
    while True:
        while (not transcriptor.push(raw_data, pause_type="small",last_block=is_last_processing_block)):
            raw_data = file_object.buffer_read(blocksize, dtype=dtype)
            block_index+=1
            if last_block_index-1 < block_index:
                is_last_processing_block=True
            else:
                is_last_processing_block=False
            
        transcpt += transcriptor.transcript[1]   
        transcriptor.refresh()   
        if is_last_processing_block:
            file_object.close()
            break
    # end = timeit.default_timer()
    # print('Time',end-start)
    print(transcpt) 

    
    
    
    
