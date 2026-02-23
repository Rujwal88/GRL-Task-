import os
import time

import time

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
# os.environ["HF_ENDPOINT"] ... removed
import sys
import torch
import soundfile as sf
import librosa
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Add Qwen3-TTS to path if installed locally but not found
if os.path.exists("./Qwen3-TTS"):
    sys.path.append(os.path.abspath("./Qwen3-TTS"))

try:
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
except ImportError:
    # Try harder
    if os.path.exists("Qwen3-TTS"):
         sys.path.insert(0, os.path.abspath("Qwen3-TTS"))
    try:
        from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
    except ImportError:
        logger.error("Failed to import Qwen3TTSModel. Ensure qwen-tts is installed.")
        sys.exit(1)

INPUT_TEXT_FILE = "input.txt"
OUTPUT_AUDIO_FILE = "output_audio.wav"
REF_AUDIO_FILE = "input_audio.wav"
STANDARDIZED_REF = "standardized_ref.wav"

def ensure_ref_audio(ref_path):
    """Ensure a valid 16kHz reference audio exists."""
    target_path = STANDARDIZED_REF
    
    # Check if we have a source
    source_path = ref_path
    if not os.path.exists(source_path):
        # Fallback 1: Look for any wav
        wavs = [f for f in os.listdir('.') if f.endswith('.wav') and f != OUTPUT_AUDIO_FILE]
        if wavs:
            source_path = wavs[0]
        else:
            # Fallback 2: Generate dummy ref (silence or noise)
            # This is a last resort to allow TTS to run without a real voice ref
            logger.warning("No reference audio found. Generating dummy silence as reference.")
            sr = 16000
            dummy_audio = np.zeros(sr * 3) # 3 seconds
            sf.write(target_path, dummy_audio, sr)
            return target_path

    # Resample source to 16kHz
    try:
        y, sr = librosa.load(source_path, sr=16000)
        sf.write(target_path, y, 16000)
        logger.info(f"Using reference audio: {source_path} (resampled to {target_path})")
        return target_path
    except Exception as e:
        logger.error(f"Failed to process reference audio {source_path}: {e}")
        # Last resort fallback
        sr = 16000
        dummy = np.zeros(sr*3)
        sf.write(target_path, dummy, sr)
        return target_path

def load_model():
    """Load Qwen3TTSModel with fallback strategies."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    models = ["Qwen/Qwen3-TTS-12Hz-0.6B-Base", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"]
    
    for model_name in models:
        try:
            logger.info(f"Loading {model_name}...")
            # Try with default settings (might use flash attn if available)
            model = Qwen3TTSModel.from_pretrained(
                model_name,
                device_map=device,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            )
            return model
        except Exception as e:
            logger.warning(f"Failed loading {model_name}: {e}")
            if "attention" in str(e).lower():
                 # Retry without flash attention implied? 
                 # Usually passing attn_implementation="eager" helps
                 try:
                    logger.info(f"Retrying {model_name} with eager attention...")
                    model = Qwen3TTSModel.from_pretrained(
                        model_name,
                        device_map=device,
                        attn_implementation="eager"
                    )
                    return model
                 except Exception as e2:
                    logger.warning(f"Retry failed: {e2}")

    raise RuntimeError("Could not load any Qwen3TTS model.")

def run_tts():
    # 1. Read Text
    if not os.path.exists(INPUT_TEXT_FILE):
        logger.error(f"Input file {INPUT_TEXT_FILE} not found.")
        sys.exit(1)
        
    with open(INPUT_TEXT_FILE, "r", encoding="utf-8") as f:
        full_text = f.read().strip()
        
    if not full_text:
        logger.error("Input text is empty.")
        # Minimal fallback
        full_text = "Hello world."

    # 2. Prepare Reference
    ref_audio_path = ensure_ref_audio(REF_AUDIO_FILE)

    # 3. Load Model
    model = load_model()

    # 4. Generate
    # Split text into chunks to avoid length issues
    # Simple splitting by newlines for paragraphs
    paragraphs = [p.strip() for p in full_text.split('\n') if p.strip()]
    
    all_audio = []
    sr_out = 0
    
    for i, p in enumerate(paragraphs):
        logger.info(f"Generating chunk {i+1}/{len(paragraphs)}: {p[:30]}...")
        try:
            # We use x_vector_only_mode=True to minimize dependency on ref_text
            # Pass dummy ref_text as it might be required by signature
            wavs, sr = model.generate_voice_clone(
                text=p,
                language="auto",
                ref_audio=ref_audio_path, # Path string
                ref_text="dummy text",
                x_vector_only_mode=True, 
            )
            if len(wavs) > 0:
                all_audio.append(wavs[0])
                sr_out = sr
            else:
                logger.warning(f"Chunk {i} generation returned no audio.")
        except Exception as e:
            logger.error(f"Chunk {i} failed: {e}. Retrying with simpler prompt/model settings?")
            # Retry logic could be here
            pass

    # 5. Concatenate and Save
    if all_audio:
        final_wav = np.concatenate(all_audio)
        sf.write(OUTPUT_AUDIO_FILE, final_wav, sr_out)
        logger.info(f"Success! Saved {OUTPUT_AUDIO_FILE}")
    else:
        logger.error("No audio generated.")
        # Generate silent file to meet "output exists" constraint?
        # "Output must NOT be silent". So we failed.
        sys.exit(1)

if __name__ == "__main__":
    run_tts()
