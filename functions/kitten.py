from kittentts import KittenTTS
import os
import soundfile as sf
from functions.audio import normalize_audio
from config import DEVICE

# Disable telemetry unconditionally.
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
# Enable offline mode unless an HF token is explicitly set, so no network
# calls are attempted (prevents SOCKS proxy / socksio errors when offline).
if not os.environ.get('HF_TOKEN'):
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'

# Kitten has "mini" and "nano" variants.
def kitten_process_audio(voice, lang, text, output):
    m = KittenTTS("KittenML/kitten-tts-nano-0.2")
    audio = m.generate(text, voice)

    # Save the audio
    sf.write(output, audio, 24000)
    # Normalize the audio
    normalize_audio(output)
