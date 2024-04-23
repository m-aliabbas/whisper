import nemo.collections.asr as nemo_asr
asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name="nvidia/parakeet-ctc-1.1b")
asr_model.transcribe(['/home/idrak_ml/techdir/whisper/WTranscriptor/audios/gettysburg.wav'])