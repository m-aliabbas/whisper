import torch
import numpy as np
# from whisper1 import WhisperTranscriptorAPI 
from whisper2 import WhisperTranscriptorAPI 
# from silero1 import SileroTranscriptorAPI
import warnings
warnings.filterwarnings('ignore')


#ASR class defination
class ASR(object):
    """
    
    Whisper Model. Takes numpy wav and return transcript
    Description of functions:

    1. __init__ : Constructor: requires a config dict but also can set default values if not found in config
    2. get_transcript: Takes in a torch array and returns the transcript
    
    """
    def __init__(self, config):
        '''
        ;Loading the vad and whisper model
        
        args:
        ;config (dictionary) having configuration for model loadings
        '''
        # configuring configs
        print(config)
        self.samplerate = config.get("samplerate", 16000)
        self.cuda_device = config.get("cuda_device", "cpu")
        self.check_interval = config.get("check_interval",3)
        #path of whisper-tiny.en; can be whisper-base.en openai/whisper-base.en
        self.model_path= config.get("model_path","tiny.en") 
        print("[INFO] Loading Models")
        self.model = WhisperTranscriptorAPI(model_path=self.model_path,)
        # self.model = SileroTranscriptorAPI()
        print("[INFO] Model Loaded")
    
    def get_transcript(self, data_torch, is_greedy=False, emissions_only=False):
        '''
         This function will generate transcripts using whisper.
         
        '''
        wave=data_torch # get the wave data
        transcript,ids = self.model.generate_transcript_numpy(wave=wave)
        #ids are model generated tokens id for the ASR
        return ids,transcript
    

if __name__ == "__main__":
    config = dict()
    config["cuda_device"] = "cpu"
    asr=ASR(config=config)
    