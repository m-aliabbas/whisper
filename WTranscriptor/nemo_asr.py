import nemo.collections.asr as nemo_asr
from transformers import pipeline
import torch
import timeit
import numpy as np
import warnings
import os
import zlib
warnings.filterwarnings('ignore')

'''
Faster Implementation of Whisper
'''


class NemoTranscriptorAPI:
    '''
    This Module is based on CTC fast Whisper for Audio Transcription.
    We need WhisperProcessor and WhisperConditionalGeneration for 
    CTC task i.e. ASR. 
    example:
          whisper_transcriptor=WhisperTranscriptorAPI(model_path='openai/whisper-tiny.en')
          
    '''
    #----------------------- constructor -------------------------------------
    #
    def __init__(self,model_path='',file_processing=False,word_timestamp=True,mac_device=False,
                 dtype = torch.float16,en_flash_attention = False,batch_size=128):

        '''
        1) Defining processor for processing audio input for Whisper and
        generate Pytorch token
        2) Put Processed Audio to model and get PyTorch Tensor later this
        tensor will be post processed by Lang Model.
        args:
          model_path: the huggingface repo of whisper-model. 
          ... i.e. for example: openai/whisper-tiny.en 
        '''
        self.model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name="nvidia/parakeet-ctc-1.1b")
        self.model_type = 'nemo'
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(device == "cuda" , "cuda check")
        
    async def genereate_transcript_from_file(self,file_name):
        t1 = timeit.default_timer()
        # speech_timestamps = get_speech_timestamps(wave, vad_model, sampling_rate=16000,threshold=0.1)
        speech_timestamps = True
        print(speech_timestamps)
        if speech_timestamps:
            wave = wave / np.iinfo(np.int16).max #normalize
            t1 = timeit.default_timer()
            if not isinstance(file_name,list):
                file_name = [file_name]
            outputs = self.model.transcribe(file_name)
            transcription = outputs[0]
            t2 = timeit.default_timer()
            print('Time taking for response',t2-t1)
            print('Audio Length',len(wave)/16000)
            return transcription,[]
        else:
            return "",[]
    #-------------------- generate transcript from nmpy array ----------------
    #
    async def generate_transcript_numpy(self, wave):
        
        '''
        Generate transcript usign a numpy array given as inpuy 
        '''
        return 'Method Not Implemented,For nemo use genereate_transcript_from_file method',[]
        t1 = timeit.default_timer()
        # speech_timestamps = get_speech_timestamps(wave, vad_model, sampling_rate=16000,threshold=0.1)
        speech_timestamps = True
        print(speech_timestamps)
        if speech_timestamps:
            wave = wave / np.iinfo(np.int16).max #normalize
            t1 = timeit.default_timer()
            file_name = await self.save_file(wave=wave)
            outputs = self.model.transcribe(['/home/idrak_ml/techdir/whisper/WTranscriptor/audios/gettysburg.wav'])
            transcription = outputs[0]
            t2 = timeit.default_timer()
            print('Time taking for response',t2-t1)
            print('Audio Length',len(wave)/16000)
            return transcription,[]
        else:
            return "",[]
