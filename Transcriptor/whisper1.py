"""Importing Libs"""

import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset
import torchaudio
import os
import numpy as np
from pydub import AudioSegment
from scipy.io.wavfile import write

"""
**WhisperTranscriptorAPI Defination**
"""

class WhisperTranscriptorAPI:
    '''
    This Module is based on OpenAI Whisper for Audio Transcription.
    We need WhisperProcessor and WhisperConditionalGeneration for 
    CTC task i.e. ASR. 
    example:
          whisper_transcriptor=WhisperTranscriptorAPI(model_path='openai/whisper-tiny.en')
          
    '''
    #----------------------- constructor -------------------------------------
    #
    def __init__(self,model_path=''):

        '''
        1) Defining processor for processing audio input for Whisper and
        generate Pytorch token
        2) Put Processed Audio to model and get PyTorch Tensor later this
        tensor will be post processed by Lang Model.
        args:
          model_path: the huggingface repo of whisper-model. 
          ... i.e. for example: openai/whisper-tiny.en 
        '''

        self.model_path = model_path
        self.processor = WhisperProcessor.from_pretrained(self.model_path) 
        self.model = WhisperForConditionalGeneration.from_pretrained(self.model_path)
        self.model.config.forced_decoder_ids = None
        self.model.config.suppress_tokens = []
        self.OUTPUT_DIR= "temp"


    #--------------------------- convert mp3 to wav ---------------------------
    #

    def convert_to_wav(self,filepath=''):
        '''
        If file is mp3 first convert it to wav.
        
        1) Read audio segment from Path
        2) Sample the audio to 16kHZ
        3) Save file in temp
        args:
            filepath(str): path of audio file example audio.mp3
        returns:
            path of temporary generated wav file
        '''

        path =filepath
        filename=os.path.basename(filepath) #Getting filename from path
        save_path = f"{self.OUTPUT_DIR}"
        if not os.path.exists(save_path): #If path not exist create a directory
            os.makedirs(save_path, exist_ok=True)
        if os.path.exists(path):
            try:
                sound = AudioSegment.from_mp3(path) #Read a sound file
                sound = sound.set_frame_rate(16000) #resample to 16000
                sound.export(f"{save_path}/{filename[:-4]}.wav", format="wav") #save to *.wav
                return f"{save_path}/{filename[:-4]}.wav"
            except:
                print(path)


    #------------------------- genrate transcript -----------------------------
    #
    def generate_transcript(self, audio_path=''):
        
        '''
        Generate the transcript from audio file using Whisper
        1) This function will read the .wav audio using torchaudio
        2) Process the audio using WhisperProcessor
        3) Generate the predicted token ids using Whisper Model
        4) Decode the output using Language model
        args: 
            audio_path (str) : path of wav file
        returns:
            transcription(str): generated script
        '''
        self.audio_path = audio_path
        exten = os.path.splitext(self.audio_path)[1]
        if exten == '.mp3':
            self.audio_path = self.convert_to_wav(self.audio_path)
        wave,fs=torchaudio.load(self.audio_path)
        #tensor to numpy
        wave=wave.numpy()
        if len(wave.shape)>=2:
            wave=wave[0,:]
        inputs = self.processor(wave, return_tensors="pt")
        input_features = inputs.input_features
        generated_ids = self.model.generate(inputs=input_features)
        #decode the transcript using language model from processor 
        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return transcription,generated_ids

    #---------------------------- save audio-----------------------------------
    #
    def save_audio(self,wave):
        
        '''
        Save the audio file for testing
        '''
        samplerate = 16000
        write("exame1.wav", samplerate, wave.astype(np.int16))
    #-------------------- generate transcript from nmpy array ----------------
    #
    
    def generate_transcript_numpy(self, wave):
        
        '''
        Generate transcript usign a numpy array given as inpuy 
        '''
        wave = wave / np.iinfo(np.int16).max #normalize
        print(wave.shape)
        self.save_audio(wave=wave)
        inputs = self.processor(wave, return_tensors="pt") #tokenize
        input_features = inputs.input_features
        generated_ids = self.model.generate(inputs=input_features)
        #decode the transcript using language model from processor 
        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return transcription,generated_ids

if __name__ == "__main__":
    """Model Initialization"""

    whisper_transcriptor=WhisperTranscriptorAPI(model_path='openai/whisper-base.en')

    """Experiments:"""

    transcript1,ids=whisper_transcriptor.generate_transcript(audio_path='delme_rec_unlimited_ykwxy3rm.wav')

    print('Transcript',transcript1)
    print('IDs Shape',ids.shape)

    #transcript2=whisper_transcriptor.genrate_transcript(audio_path='abc/audio5.wav')

    #print(transcript2)

    #transcript3=whisper_transcriptor.genrate_transcript(audio_path='ali.mp3')

    #print(transcript3)

    #transcript4=whisper_transcriptor.genrate_transcript(audio_path='temp/ali.wav')

    #print(transcript4)
