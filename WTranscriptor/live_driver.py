import sounddevice as sd
from KTranscriptor import KTranscriptor
SAMPLE_RATE = 16000
CHUNK = 4000  # 0.25 seconds per chunk

def main():
    transcriptor = KTranscriptor()

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1) as stream:
        print("Recording... Press Ctrl+C to stop.")
        try:
            while True:
                audio_chunk, overflowed = stream.read(CHUNK)
                if transcriptor.push(audio_chunk,pause_length=1):
                    print("Transcription: ", transcriptor.transcript)
        except KeyboardInterrupt:
            pass

    print("Final transcript:", transcriptor.transcript)

if __name__ == '__main__':
    main()