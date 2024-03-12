# transcriptor_module
Changes:

1) Wave2Vec/HuBert is replaced with WhisperASR 
2) For Pause Detection SileroVAD https://pytorch.org/hub/snakers4_silero-vad_vad/
is used.
3) Pauses are detected inside Transcriptor.push functions using vad.
4) Transcript are generated on 3s pause or 10s completion.


# To run server

Goto WTTranscriptor
Run the command ```uvicorn server:app --host IP_ADDRESS --port PORT_NUMBER --reload ```

Usage:

    1 ) 
    Usage whisper1.py when you have to apply it directly on longer file 30s or more.
    It will work fast. Also, make file_processing flag to `True`

    `whisper_transcriptor=WhisperTranscriptorAPI(model_path='openai/whisper-base.en',file_processing=True)`


    2) 
    Please Use live_driver.py when you have to try in live mic input.



 
