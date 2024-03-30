import numpy as np
import librosa
import os

def delete_file_if_exists(file_path):
    """Deletes the file at file_path if it exists."""
    if os.path.exists(file_path):
        os.remove(file_path)
        # print(f"File '{file_path}' has been deleted.")
    else:
        # print(f"No file found at '{file_path}'. Nothing to delete.")
        pass


def convert_string_to_bytes(input_str: str) -> bytes:
    try:
        audio_bytes = eval(input_str)
        if not isinstance(audio_bytes, bytes):
            raise ValueError("Input string does not evaluate to bytes.")
        return audio_bytes
    except SyntaxError as e:
        raise ValueError("Invalid input string format.") from e
    
def bytes_to_numpy(audio_bytes: bytes) -> np.ndarray:
    return np.frombuffer(audio_bytes, dtype=np.int16)

def read_wav_as_int16(file_path, target_sr=16000):
    # Load the WAV file with librosa
    audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
    
    # Convert from float (range -1 to 1) to int16 (range -32768 to 32767)
    audio_int16 = np.int16(audio * 32767)
    
    return audio_int16, sr