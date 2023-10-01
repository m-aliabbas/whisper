# -- Configurations for the Transcriptor module
import torch
from WTranscriptor import enums as Trans_enums
config = dict()


CLASSIFIER_PATH = '/home/idrak/ali_own/farooq/'

# -------------- General configs ------------#
config["samplerate"] = 16000
config["cuda_device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config["local_flag"] = True # running on local machine or not

# -------------- Transcriptor configs ------------#
config["maximum_allowable_duration"] = 5 # max duration of input audio to transcribe in secon
config["duration_threshold_delta"] = 1 # increase in duration thresold, for next iteration. 
config["pause_time"] =  2 # time of silence identified as a large pause   (seconds)
config["maximum_noresponse_duration"] = config["pause_time"] * 4  # max duration at the start of audio after which a no response is detected, should be greater than pause time

# -------------- Wav2Vec model configs ------------#
config["one_sec_chunk_size"] =  50 # Chunck of Wav2Vec emissions equal to one second. The value can change with samplerate. For 16000 samplerate, 
                                    # 50 chunks is one second. 
config["decoder_type"] =   "beam"   #currently supports beam and greedy

#--------------- Whisper ASR model configs ----------------#
config["model_path"] =   "openai/whisper-small.en"

# -------------- Decoder configs ------------#
if config["decoder_type"] == "beam":
    config["decoder"] = dict()
    config["decoder"]["LM_WEIGHT"] =  2#3.23 # weight given to the language model in decoding.
    config["decoder"]["WORD_SCORE"] =  -0.26 # score added at the end of decoding
    config["decoder"]["BEAM_SIZE"] =  100#500 # Number of paralell hypothesis that a decoder can run and choose the best
else:
    config["decoder"] = dict()
