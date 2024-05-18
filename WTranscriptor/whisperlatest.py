from transformers import pipeline,WhisperProcessor, WhisperForConditionalGeneration,WhisperTokenizer
import torch
import timeit
import numpy as np
import warnings
import os
import zlib
from typing import Optional,Dict,List,Collection
warnings.filterwarnings('ignore')
from utils.utils import *
'''
Faster Implementation of Whisper
'''

# torch.set_num_threads(8)


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
    def detect_language(self,model: WhisperForConditionalGeneration, tokenizer: WhisperTokenizer, input_features,
                    possible_languages: Optional[Collection[str]] = None) -> List[Dict[str, float]]:
        # hacky, but all language tokens and only language tokens are 6 characters long
        language_tokens = [t for t in tokenizer.additional_special_tokens if len(t) == 6]
        # if possible_languages is not None:
        #     language_tokens = [t for t in language_tokens if t[2:-2] in possible_languages]
        #     if len(language_tokens) < len(possible_languages):
        #         raise RuntimeError(f'Some languages in {possible_languages} did not have associated language tokens')

        language_token_ids = tokenizer.convert_tokens_to_ids(language_tokens)

        # 50258 is the token for transcribing
        logits = model(input_features,
                    decoder_input_ids = torch.tensor([[50258] for _ in range(input_features.shape[0])])).logits
        mask = torch.ones(logits.shape[-1], dtype=torch.bool)
        mask[language_token_ids] = False
        logits[:, :, mask] = -float('inf')

        output_probs = logits.softmax(dim=-1).cpu()
        return [
            {
                lang: output_probs[input_idx, 0, token_id].item()
                for token_id, lang in zip(language_token_ids, language_tokens)
            }
            for input_idx in range(logits.shape[0])
        ]
    def language_detection(self,wave):
        processor = WhisperProcessor.from_pretrained('openai/whisper-tiny.en')
        model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-tiny.en')
        tokenizer = WhisperTokenizer.from_pretrained('openai/whisper-tiny.en')
        input_features = processor(wave, sampling_rate=16000,
                               return_tensors="pt").input_features

        language = self.detect_language(model, tokenizer, input_features, {'en', 'zh'})
        print(language)

    def __init__(self,model_path='',file_processing=False,word_timestamp=True,mac_device=False,
                 dtype = torch.float16,en_flash_attention = False,batch_size=128,
                 vad_model = None):

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
        # print(self.model_path)
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
        self.vad_model, self.utils = torch.hub.load('snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=False)

        (self.get_speech_timestamps,
        self.save_audio,
        self.read_audio,
        self.VADIterator,
        self.collect_chunks) = self.utils

        
    
    #-------------------- generate transcript from nmpy array ----------------
    #
    
    async def generate_transcript_numpy(self, wave,sample_rate=16000):
        
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
<<<<<<< HEAD
        with torch.no_grad():
            speech_timestamps = self.get_speech_timestamps(wave, self.vad_model, sampling_rate=16000,threshold=0.5)
        # speech_timestamps = True
        print(speech_timestamps)
        if speech_timestamps:
            wave = torch.from_numpy(wave)
            if len(speech_timestamps) <=0 :
                print('NO sound')
                return '', []

            wave1 = self.collect_chunks(speech_timestamps, wave)
            wave = wave1.numpy()
=======
        # speech_timestamps = get_speech_timestamps(wave, vad_model, sampling_rate=16000,threshold=0.1)
        speech_timestamps = True
        # print(speech_timestamps)
        if speech_timestamps:
            # wave = torch.from_numpy(wave)
            # wave1 = collect_chunks(speech_timestamps, wave)
            # wave = wave1.numpy()

>>>>>>> b8ffaeb (Sampling issue resolved)
            wave = wave / np.iinfo(np.int16).max #normalize
            # self.language_detection(wave)
            t1 = timeit.default_timer()
            outputs = self.model(
                wave,
            chunk_length_s=15,
            batch_size=self.batch_size,
            generate_kwargs=generate_kwargs,
            return_timestamps=False,
                 )
            transcription = filter_hallucination(outputs['text'])
            print(transcription)
            t2 = timeit.default_timer()
            print('Time taking for response',t2-t1)
            print('Audio Length',len(wave)/16000)
            return transcription,[]
        else:
            return "",[]
    async def genereate_transcript_from_file(self, file_name):
        return 'method not implemented; for whisper use generate_transcript_numpy', []
