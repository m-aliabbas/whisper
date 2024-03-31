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


vad_model, utils = torch.hub.load('snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=False)


(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils

class WhisperTranscriptorAPI:
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
                 dtype = torch.float16,en_flash_attention = False,batch_size=32):

        '''
        1) Defining processor for processing audio input for Whisper and
        generate Pytorch token
        2) Put Processed Audio to model and get PyTorch Tensor later this
        tensor will be post processed by Lang Model.
        args:
          model_path: the huggingface repo of whisper-model. 
          ... i.e. for example: openai/whisper-tiny.en 
        '''
        self.mac_device = mac_device
        self.model_path = model_path
        self.batch_size = batch_size
        print(self.model_path)
        # device='cpu'
        # compute_type="int8"
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(device == "cuda" , "cuda check")
        if mac_device:
            print(f"[INFO] Loading on Mac Device")
            self.model = pipeline(
                    "automatic-speech-recognition",
                    model=self.model_path,
                    torch_dtype=torch.float16,
                    device="mps" if mac_device else f"cuda:{cuda_device_id}",
                )
        else:
            if device == 'cuda':
                cuda_device_id = 0
                print(f"[INFO] Loading {self.model_path} on Cuda")
                try:
                    self.model = pipeline(
                    "automatic-speech-recognition",
                    model=self.model_path,
                    torch_dtype=torch.float16,
                    device="mps" if mac_device else f"cuda:{cuda_device_id}",
                    model_kwargs={"attn_implementation": "flash_attention_2"} if en_flash_attention else {"attn_implementation": "sdpa"},
                        )
                except ValueError:
                    print("[INFO] Cuda Support Issue Moving to CPU")
                    self.model = pipeline(
                    "automatic-speech-recognition",
                    model=self.model_path,
                    torch_dtype=torch.float16,
                    device=device)
            else:
                    print("[INFO] Loading on CPU")
                    self.model = pipeline(
                    "automatic-speech-recognition",
                    model=self.model_path,
                    torch_dtype=torch.float16,
                    device=device)
        self.OUTPUT_DIR= "audios"

    #-------------------- generate transcript from nmpy array ----------------
    #
    
    def generate_transcript_numpy(self, wave):
        
        '''
        Generate transcript usign a numpy array given as inpuy 
        '''

        if self.mac_device:
            torch.mps.empty_cache()
        generate_kwargs = {"task": 'transcribe', "language": 'en'}
        if self.model_path.split(".")[-1] == "en":
            generate_kwargs.pop("task")
            generate_kwargs.pop("language") 
        
        t1 = timeit.default_timer()
        #speech_timestamps = get_speech_timestamps(wave, vad_model, sampling_rate=16000,threshold=0.1)
        speech_timestamps = True
        print(speech_timestamps)
        if speech_timestamps:
            #wave = torch.from_numpy(wave)
            # wave1 = collect_chunks(speech_timestamps, wave)
            # wave = wave1.numpy()
            wave = wave / np.iinfo(np.int16).max #normalize
            t1 = timeit.default_timer()
            outputs = self.model(
                wave,
            chunk_length_s=30,
            batch_size=self.batch_size,
            generate_kwargs=generate_kwargs,
            return_timestamps=False,
                 )
            transcription = outputs['text']
            t2 = timeit.default_timer()
            print('Time taking for response',t2-t1)
            print('Audio Length',len(wave)/16000)
            return transcription,[]
        else:
            return "",[]


