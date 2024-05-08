import numpy as np
import librosa
import os

from concurrent.futures import ThreadPoolExecutor
import soundfile as sf
import numpy as np
import uuid
import asyncio

suppress_low = [
    "Thank you",
    "Thanks for",
    "ike and ",
    "lease sub",
    "The end.",
    "ubscribe",
    "my channel",
    "the channel",
    "our channel",
    "ollow me on",
    "for watching",
    "hank you for watching",
    "for your viewing",
    "r viewing",
    "Amara",
    "next video",
    "full video",
    "ranslation by",
    "ranslated by",
    "ee you next week",
    "video",
    "See you, bye-bye.",
    'bye',
    'bye-bye',
    'See you, bye-bye.',
    '..',
    'hhhh',


    
]

def filter_hallucination(transcript):
    for token  in suppress_low:
        if token in transcript:
            return ''
    hal = ['you','your','video','thank']
    if len(transcript) < 6:
        for hal_st in hal:
            if hal_st in transcript:
                return ''
    return transcript
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

# Create a global thread pool executor
executor = ThreadPoolExecutor()

async def async_save_wav(numpy_array, sample_rate=16000):
    """
    Asynchronously saves a NumPy array as a mono WAV file with a specified sample rate using ThreadPoolExecutor.
    
    Args:
        numpy_array (np.ndarray): The audio data to save.
        sample_rate (int): The sample rate of the audio file.

    Returns:
        str: The name of the created WAV file.
    """
    filename = f"temp/{uuid.uuid4()}.wav"

    # Offload the blocking operation to the thread pool
    await asyncio.get_running_loop().run_in_executor(
        executor, 
        save_wav_sync,  # This is the synchronous save function
        numpy_array, 
        sample_rate, 
        filename
    )

    return filename

def save_wav_sync(numpy_array, sample_rate=16000, filename = f"temp/{uuid.uuid4()}.wav"):
    """
    Synchronously saves a NumPy array as a mono WAV file.
    This function is intended to be run in a ThreadPoolExecutor.
    """
    filename = f"temp/{uuid.uuid4()}.wav"
    sf.write(filename, numpy_array, sample_rate)
    return filename

