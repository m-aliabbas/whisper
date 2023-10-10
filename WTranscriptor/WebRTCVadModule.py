import webrtcvad
from collections import deque
import itertools
import time  # Import the time module

# Assuming `enums` contains the necessary enumerations for PAUSE and NO_PAUSE
# If not, you should define them yourself.
import enums

class WebRTCVadModule:
    '''
    This class is an interface for Voice Activity Detection based on WebRTC VAD.
    '''

    def __init__(self, config=dict()) -> None:
        '''
        Args:
        config (dict): Configuration dictionary having parameters:
                      - sample_rate: Sampling rate of audio (default: 16000)
                      - frame_duration: Duration of each frame in ms (default: 30ms)
                      - pause_threshold: Number of silent frames to consider as a pause
                      - buffer_size: Size of the VAD decision buffer
                      - min_speech_count: Minimum number of continuous speech frames to consider as real speech
        '''
        self.SAMPLE_RATE = config.get("sample_rate", 16000)
        self.FRAME_DURATION = config.get("frame_duration", 30)
        self.FRAME_SIZE = int(self.SAMPLE_RATE * self.FRAME_DURATION / 1000)
        
        # VAD decision buffer and parameters
        self.PAUSE_THRESHOLD = config.get("pause_threshold", 50)
        self.BUFFER_SIZE = config.get("buffer_size", 20)
        self.MIN_SPEECH_COUNT = config.get("min_speech_count", 15)
        
        self.vad_buffer = deque([0] * self.BUFFER_SIZE, maxlen=self.BUFFER_SIZE)
        self.vad = webrtcvad.Vad(2)
        self.silent_frames = 0
        self.start_time = None
        self.speech_detected = False

        print('Inside WebRTC Vad')

    def is_speech_segment_too_short(self) -> bool:
        """Check if the current speech segment in the buffer is shorter than the threshold."""
        speech_segments = [list(group) for val, group in itertools.groupby(self.vad_buffer) if val == 1]
        if not speech_segments:
            return True
        return len(speech_segments[-1]) < self.MIN_SPEECH_COUNT

    def reset_start_time(self) -> None:
        """Reset the start time for speech detection."""
        self.start_time = None

    def update_start_time(self) -> None:
        """Set the start time for speech detection if not set."""
        if self.start_time is None:
            self.start_time = time.time()  # Get the current time in seconds since the epoch

    def refresh(self) -> None:
        """Reset the buffer and silent frames counter."""
        self.silent_frames = 0
        self.vad_buffer = deque([0] * self.BUFFER_SIZE, maxlen=self.BUFFER_SIZE)
        print('Listening from Again')

    def get_pause_status(self, data) -> str:
        """Check voice activity and return pause status based on the duration threshold."""
        
        current_time = time.time()  # Get the current time in seconds since the epoch
        
        # If user did not speak in the first 1.5 sec
        if not self.speech_detected:
            self.update_start_time()  # Using the updated method
            # If more than 1.5 seconds has passed since start_time without speech
            if current_time - self.start_time > 1.5:
                self.reset_start_time()
                return enums.NO_RESPONSE

            is_speech = self.vad.is_speech(data.tobytes(), sample_rate=self.SAMPLE_RATE)    
            print('Is speech',is_speech)
            self.vad_buffer.append(is_speech)  # Storing this decision in the buffer for checking segment length

            # Check if the segment is too short or there's no speech
            if not is_speech or self.is_speech_segment_too_short():
                return enums.GUESTURE_OF_LISTENING
            else:
                self.speech_detected = True

        else:
            is_speech = self.vad.is_speech(data.tobytes(), sample_rate=self.SAMPLE_RATE)
            self.vad_buffer.append(is_speech)

        if is_speech:
            self.silent_frames = 0
        else:
            self.silent_frames += 1

        if is_speech and self.is_speech_segment_too_short():
            # Pause Detected and we are sending PAUSE;
            # Enums are inverted so
            self.refresh()
            return enums.NO_PAUSE

        if self.silent_frames >= self.PAUSE_THRESHOLD:
            # Pause Detected and we are sending PAUSE;
            # Enums are inverted so
            self.refresh()
            return enums.NO_PAUSE
        #Detecting speech; Sorry its inverted
        return enums.PAUSE

