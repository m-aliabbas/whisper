from SileroVadModule import SileroVadModule

class VAD_Interface(object):
    def __init__(self,config) -> None:
        '''
        An Interface Between Transcriptor and VAD Module
        '''
        self.config = config
        self.vad = SileroVadModule(config=self.config)
    def pause_status(self,data):
        '''
        return pause based on specified VAD Module
        '''
        return self.vad.get_pause_status(data=data)
