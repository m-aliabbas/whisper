import numpy as np
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