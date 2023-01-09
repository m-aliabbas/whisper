# transcriptor_module
Changes:

1) Wave2Vec/HuBert is replaced with WhisperASR 
2) For Pause Detection SileroVAD https://pytorch.org/hub/snakers4_silero-vad_vad/
is used.
3) Pauses are detected inside Transcriptor.push functions using vad.
4) Transcript are generated on 3s pause or 10s completion.


 
