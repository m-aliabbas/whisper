import torch
import numpy as np
import soundfile as sf
from WhisperASR import ASR
import enums
import sounddevice as sd
vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=False)

(get_speech_timestamps,
 _, read_audio,VADIterator,
 *_) = utils


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
        self.config = config
        self.asr = ASR(config)
        self.vad_model= vad_model
        self.vad_threshold=config.get("vad_threshold",0.6) 
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

    def push(self, raw_audio_block, pause_type=1, is_greedy=False, verbose=False):
        """
        The main function of the Transcriptor module. For an example usage, see the code in __main__ in Transcriptor.py

        Args:
            raw_audio_block: Output from sounddevice raw stream of arbitrary blocksizey.
            
        """

        #obtaining audio from bytes
        tmp_np = self.byte2np(raw_audio_block)
        self.data_array = np.hstack((self.data_array,tmp_np))
        duration  = len(self.data_array) / self.samplerate #duration 16000/16000=1s


        if duration >= self.duration_threshold: #if duration is larger than 3s
            data = self.data_array
            #passing data from VAD Model
            speech_dict = get_speech_timestamps(data, vad_model, sampling_rate=int(self.samplerate),threshold=self.vad_threshold)
            # print(speech_dict)
            if speech_dict: #if speech detected
                max_end = max(speech_dict, key=lambda x:x['end'])  #checking the end of speech
                #if current data frame and last speech index gap is larger than 48000 i.e. 3 sec
                if ((len(data)-max_end['end'])/self.samplerate) >= self.duration_threshold: #small pause detected;
                    
                    print('[+] Pause Detected')
                    self.status=True
            

            if duration > self.max_allowable_duration: #10 second acheived
                print("[-] Max Limit Exceed")
                self.status = True
            
            transcript=''
            #generate transcript
            ids,transcript = self.asr.get_transcript(data)
            self.transcript = transcript
            # print(transcript)
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

    filepath  = "/home/ali/Desktop/transcriptor/s1.wav"
    file_object =  sf.SoundFile(filepath)
    blocksize = 8000
    dtype = 'int16'
    samplerate = file_object.samplerate
    config=dict()
    transcriptor = WTranscriptor()
    raw_data = file_object.buffer_read(blocksize, dtype=dtype)
    while (not transcriptor.push(raw_data, pause_type="small")):
        raw_data = file_object.buffer_read(blocksize, dtype=dtype)
        if len(raw_data) == 0:
            break
        
    print(transcriptor.transcript)
    transcriptor.refresh()
        
    
    
    
    
