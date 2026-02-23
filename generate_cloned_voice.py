import os
import sys
import torch
import soundfile as sf
import speech_recognition as sr
import logging
import re
import numpy as np
from pydub import AudioSegment

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def transcribe_audio(audio_path):
    logger.info(f"Transcribing reference audio: {audio_path}")
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            logger.info(f"Reference Transcription: {text}")
            return text
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return None

def split_text(text, max_chars=300):
    # Split by sentence endings but try to keep chunks under max_chars
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""
    for s in sentences:
        if len(current_chunk) + len(s) < max_chars:
            current_chunk += " " + s
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = s
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def main():
    try:
        from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
    except ImportError:
        logger.error("qwen_tts not found. Please ensure it's installed.")
        return
    
    input_text_file = "input.txt"
    ref_audio_file = "standardized_ref.wav"
    output_audio_file = "output_audio.wav"
    
    if not os.path.exists(input_text_file):
        logger.error(f"{input_text_file} not found!")
        return

    with open(input_text_file, "r", encoding="utf-8") as f:
        full_text = f.read().strip()
    
    if not full_text:
        logger.error("Input text is empty!")
        return

    # Check for ref audio
    if not os.path.exists(ref_audio_file):
        wavs = [f for f in os.listdir('.') if f.endswith('.wav') and f != output_audio_file]
        if wavs:
            ref_audio_file = wavs[0]
            logger.warning(f"Using {ref_audio_file} as reference.")
        else:
            logger.error("No reference audio found!")
            return

    # Transcribe ref audio
    ref_text = transcribe_audio(ref_audio_file)
    x_vector_only = ref_text is None

    logger.info("Loading Qwen3 TTS Model...")
    # Model selection - 0.6B is safer for environments with limited resources
    model_name = "Qwen/Qwen3-TTS-12Hz-0.6B-Base" 
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    try:
        # Avoid device_map="auto" to prevent disk offload issues
        model = Qwen3TTSModel.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch.float32 if device == "cpu" else torch.bfloat16
        )
        logger.info(f"Model {model_name} loaded on {device}.")
    except Exception as e:
        logger.warning(f"Failed to load on {device}: {e}. Retrying with CPU and float32...")
        try:
            model = Qwen3TTSModel.from_pretrained(
                model_name,
                device_map="cpu",
                torch_dtype=torch.float32
            )
            logger.info(f"Model {model_name} loaded on CPU.")
        except Exception as e2:
            logger.error(f"Failed to load model on CPU: {e2}")
            return

    chunks = split_text(full_text)
    logger.info(f"Split text into {len(chunks)} chunks.")
    
    all_wavs = []
    sample_rate = 16000 # Default fallback
    
    # Reusable prompt to save time
    try:
        prompt_items = model.create_voice_clone_prompt(
            ref_audio=ref_audio_file,
            ref_text=ref_text if not x_vector_only else None,
            x_vector_only_mode=x_vector_only
        )
    except Exception as e:
        logger.error(f"Failed to create prompt: {e}")
        return

    for i, chunk in enumerate(chunks):
        logger.info(f"Generating chunk {i+1}/{len(chunks)} ({len(chunk)} chars)...")
        try:
            wavs, sr_out = model.generate_voice_clone(
                text=chunk,
                language="Auto",
                voice_clone_prompt=prompt_items
            )
            if wavs:
                all_wavs.append(wavs[0])
                sample_rate = sr_out
        except Exception as e:
            logger.error(f"Failed generating chunk {i}: {e}")
            continue
    
    if all_wavs:
        # Concatenate numpy arrays
        final_wav = np.concatenate(all_wavs)
        sf.write(output_audio_file, final_wav, sample_rate)
        logger.info(f"Successfully saved cloned voice to {output_audio_file}")
    else:
        logger.error("No audio generated.")

if __name__ == "__main__":
    main()
