import os
import sys
import torch
import soundfile as sf
import librosa
import numpy as np
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

INPUT_TEXT_FILE = "input.txt"
OUTPUT_AUDIO_FILE = "output_audio.wav"
REF_AUDIO_FILE = "input_audio.wav"
STANDARDIZED_REF = "standardized_ref.wav"

def get_model():
    """Import and load model with robust error handling."""
    try:
        from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
    except ImportError:
        logger.error("Failed to import Qwen3TTSModel. Ensure qwen-tts is installed.")
        # Try installing if missing? No, user constraint says we should have installed it.
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # Try 0.6B first as it is smaller and less likely to OOM
    model_name = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    
    try:
        logger.info(f"Loading {model_name}...")
        model = Qwen3TTSModel.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        return model
    except Exception as e:
        logger.error(f"Failed to load {model_name}: {e}")
        # Build prompt to retry? Default retry is bad if download failed.
        sys.exit(1)

def chunk_text(text, max_chars=300):
    """Split text into chunks of roughly max_chars, splitting at sentence boundaries."""
    sentences = text.replace('.', '.|').replace('?', '?|').replace('!', '!|').split('|')
    chunks = []
    current_chunk = ""
    
    for s in sentences:
        if len(current_chunk) + len(s) < max_chars:
            current_chunk += s
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = s
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def main():
    if not os.path.exists(INPUT_TEXT_FILE):
        logger.error("Input file missing.")
        sys.exit(1)
        
    with open(INPUT_TEXT_FILE, "r", encoding="utf-8") as f:
        text = f.read().strip()
        
    if not text:
        logger.error("Input text empty.")
        sys.exit(1)
        
    # Standardize ref
    if not os.path.exists(STANDARDIZED_REF):
        # Create dummy or use existing
        if os.path.exists(REF_AUDIO_FILE):
             y, sr = librosa.load(REF_AUDIO_FILE, sr=16000)
             sf.write(STANDARDIZED_REF, y, 16000)
        else:
             logger.warning("No ref audio. Using silence.")
             sf.write(STANDARDIZED_REF, np.zeros(16000*3), 16000)
             
    model = get_model()
    
    # Process
    chunks = chunk_text(text)
    audio_segments = []
    
    for i, c in enumerate(chunks):
        logger.info(f"Processing chunk {i+1}/{len(chunks)}: {c[:20]}...")
        try:
            # x_vector_only_mode=True means extracting voice from audio, ignoring ref_text
            wavs, sr = model.generate_voice_clone(
                text=c,
                ref_audio=STANDARDIZED_REF,
                ref_text="dummy", 
                x_vector_only_mode=True,
                language="auto"
            )
            if len(wavs) > 0:
                audio_segments.append(wavs[0])
            else:
                logger.warning(f"Chunk {i+1} returned no audio.")
        except Exception as e:
            logger.error(f"Chunk {i+1} failed: {e}")
            
    if audio_segments:
        final_audio = np.concatenate(audio_segments)
        sf.write(OUTPUT_AUDIO_FILE, final_audio, sr) # model sr usually 16000 or 24000
        logger.info(f"Saved {OUTPUT_AUDIO_FILE}, duration: {len(final_audio)/sr:.2f}s")
    else:
        logger.error("No audio generated.")
        sys.exit(1)

if __name__ == "__main__":
    main()
