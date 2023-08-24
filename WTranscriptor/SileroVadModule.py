import torch
import audioop
import os
#------------------- For Voice Activity Detection Model Loading --------------
#

# /home/ali/.cache/torch/hub/snakers4_silero-vad_master
os.environ['TORCH_HOME'] = '/home/mohammadali/.cache/torch'
# vad_model, utils = torch.hub.load(repo_or_dir='/home/ali/.cache/torch/hub/snakers4_silero-vad_master',
#                                   source= 'local',
#                               model='silero_vad',
#                               force_reload=False)

vad_model, utils = torch.hub.load('snakers4/silero-vad',
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
        self.vad_threshold=config.get("vad_threshold",0.1)
        self.duration_threshold = config.get("duration_threshold", 0.8)
        self.audio_samples = []
        self.thresold_rms = 2000
        self.min_rms = float('inf')
        self.max_rms = 0
        self.decay_factor = 0.99
        self.pause_counter = 0

        print('Inside Silero')

    def get_pause_status(self, data):
        '''
        Get the Data and Check Voice Activity. If Duration Threshold frame is 
        empty return a Pause Status of True otherwise false.
        '''
        pause_status = False
        speech_dict = None
        # Assuming you have other initializations here
        
        # Calculate the RMS for the current data segment
        current_rms = audioop.rms(data, 2)
        
        # Update the min and max RMS values
        self.min_rms = min(self.min_rms, current_rms)
        self.max_rms = max(self.max_rms, current_rms)

        # Get speech timestamps using VAD
        speech_dict = get_speech_timestamps(data, vad_model, sampling_rate=int(self.sample_rate), threshold=0.5)

        if speech_dict:  # if speech detected
            max_end = max(speech_dict, key=lambda x: x['end'])  # checking the end of speech
            
            # if current data frame and last speech index gap is larger than 48000 i.e. 3 sec
            if ((len(data) - max_end['end']) / self.sample_rate) >= self.duration_threshold:
                print("Speech Pause Detected By VAD")
                pause_status = True

            if len(data) > 30000:
                # Use RMS range to determine pause
                
                threshold = self.min_rms + (self.max_rms - self.min_rms) / 2
                if current_rms < threshold:
                    self.pause_counter += 1
                   
                else:
                    self.pause_counter = 0
                
                if self.pause_counter >= 2:  # some_value could be 2, 3, etc. depending on your desired pause length
                    pause_status = True

        else:
            if len(data) >= 32000:
                print('No Speech in 2s')
                pause_status = True  # You might adjust this to also consider RMS range or other factors
        
        # Decay the min and max RMS over time for adaptability
        self.min_rms = self.min_rms * self.decay_factor + current_rms * (1 - self.decay_factor)
        self.max_rms = self.max_rms * self.decay_factor + current_rms * (1 - self.decay_factor)
        
        return pause_status