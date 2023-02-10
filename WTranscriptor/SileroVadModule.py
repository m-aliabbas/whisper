import torch

#------------------- For Voice Activity Detection Model Loading --------------
#
vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=False)

(get_speech_timestamps,
 _, read_audio,VADIterator,
 *_) = utils

class SileroVadModule(object):
    '''

    This class is interface for Voice activity detection based on 
    Silero VAD. We are loading module from torch model hub.

    '''
    def __init__(self,config=dict()) -> None:
        '''
        vad_model: silero vad model from torch model hub
        args: 
        config(dict): a dictionary having following configuration
        sample_rate(int) : sampling rate of audio
        vad_threshold(float): probility of being speech
        duration_threshold(int): how many seconds of apuse you want to add
        '''
        self.vad_model = vad_model
        self.sample_rate = config.get("sample_rate", 16000)
        self.vad_threshold=config.get("vad_threshold",0.6)
        self.duration_threshold = config.get("duration_threshold", 3)

    def get_pause_status(self,data):
        '''
        Get the Data and Check Voice Activity. If Duration Threshold frame is 
        empty return a Pause Status of True otherwise false.
        '''
        pause_status=False
        speech_dict = None
        speech_dict = get_speech_timestamps(data, vad_model, sampling_rate=int(self.sample_rate),threshold=self.vad_threshold)
        if speech_dict: #if speech detected
            max_end = max(speech_dict, key=lambda x:x['end'])  #checking the end of speech
            #if current data frame and last speech index gap is larger than 48000 i.e. 3 sec
            if ((len(data)-max_end['end'])/self.sample_rate) >= self.duration_threshold: #small pause detected;
                pause_status = True
        return pause_status