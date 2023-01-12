import queue
import sounddevice as sd
from Transcriptor import Transcriptor
class InputStreamer():
    def __init__(self, device) -> None:
        try:
            self.q = queue.Queue()
            input_dev = device
            self.in_stream = sd.RawInputStream(samplerate=16000, blocksize = 8000, device=input_dev, dtype='int16',
                                            channels=1, callback=self.in_callback)
        except ValueError:
            print("Error! Kindly check the name of the input device (virtual cable).")
            exit()
        except Exception as e:
            print("Error!", e)
    def in_callback(self, indata, frames, time, status):
        self.q.put(bytes(indata))
    def refresh(self):
        self.q = queue.Queue()
        
def get_transcript(in_stream_obj, transcriptor):
    in_stream_obj.in_stream.start()
    transcriptor.refresh()
    rawData = in_stream_obj.q.get() # geting raw data of callee from the q for processing
    # print(rawData)
    while (not transcriptor.push(rawData, pause_type="small", verbose=False)):
        rawData = in_stream_obj.q.get()
        # print(rawData)
    result = transcriptor.transcript
    transcriptor.refresh()
    return result

input_idx = 16 #ali_system device
config=dict()
transcriptor = Transcriptor(config=config)
input_stream_obj = InputStreamer(device=input_idx)

print("[+++] Everything loaded")
transcript=get_transcript(in_stream_obj=input_stream_obj,transcriptor=transcriptor)
print("[+++] Function Completed")
print(f"[+++] {transcript}")
