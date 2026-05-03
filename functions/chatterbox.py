import os
import torch
import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from functions.audio import normalize_audio
from config import DEVICE

# Disable HuggingFace Hub and Transformers network access when no token is set,
# preventing SOCKS proxy / socksio errors in offline environments.
if not os.environ.get('HF_TOKEN'):
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

def chatterbox_process_audio(voice, lang, text, output):

    # Clear CUDA cache
    torch.cuda.empty_cache()

    model = ChatterboxMultilingualTTS.from_pretrained(device=DEVICE)
    wav = model.generate(text, language_id=lang, audio_prompt_path=voice)

    ta.save(output, wav, model.sr)

    # Save the audio
    sf.write(output, audio, 24000)
    # Normalize the audio
    normalize_audio(output)