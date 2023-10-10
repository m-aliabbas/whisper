from SileroVadModule import SileroVadModule
from WebRTCVadModule import WebRTCVadModule

class VAD_Interface(object):
    def __init__(self,config) -> None:
        '''
        An Interface Between Transcriptor and VAD Module
        '''
        print(' VAD Config',config)
        self.config = config
        # self.vad = SileroVadModule(config=self.config)
        self.vad = WebRTCVadModule()
    def pause_status(self,data):
        '''
        return pause based on specified VAD Module
        '''
        return self.vad.get_pause_status(data=data)
