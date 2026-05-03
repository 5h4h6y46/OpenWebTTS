import warnings
import os
# Suppress torch warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch.nn.modules.rnn')
warnings.filterwarnings('ignore', category=FutureWarning, module='torch.nn.utils.weight_norm')
# Suppress HuggingFace Hub warnings
warnings.filterwarnings('ignore', message='.*HF_TOKEN.*')

# Set HF_HUB_DISABLE_TELEMETRY to suppress additional warnings
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

KOKORO_MODEL_ID = 'hexgrad/Kokoro-82M'


def _is_model_cached(model_id: str) -> bool:
    """Check if a HuggingFace model is present in the local cache."""
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        model_dir = 'models--' + model_id.replace('/', '--')
        return os.path.isdir(os.path.join(HF_HUB_CACHE, model_dir))
    except Exception:
        return False


# Enable offline mode when the model is already cached to avoid unnecessary
# network requests (which can fail due to proxy issues or no internet access).
# Also fall back to offline mode when no HF_TOKEN is set (previous behaviour).
# Respect an explicitly set HF_HUB_OFFLINE environment variable.
if not os.environ.get('HF_HUB_OFFLINE'):
    if _is_model_cached(KOKORO_MODEL_ID) or not os.environ.get('HF_TOKEN'):
        os.environ['HF_HUB_OFFLINE'] = '1'

from kokoro import KPipeline
import soundfile as sf
import torch
from functions.audio import normalize_audio
from config import DEVICE

def kokoro_process_audio(voice, lang, text, output):

    # If we don't have a set lang, the first letter of the voice name will tell us.
    if lang == False:
        lang = voice[0]
    
    # Explicitly pass repo_id to suppress warning
    pipeline = KPipeline(lang, device=DEVICE, repo_id=KOKORO_MODEL_ID)
    generator = pipeline(text, voice)

    for i, (gs, ps, audio) in enumerate(generator):
        sf.write(f'{output}', audio, 24000)

    # Normalize the audio
    normalize_audio(output)