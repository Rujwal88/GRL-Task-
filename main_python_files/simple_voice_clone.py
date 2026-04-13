"""
Qwen3 TTS Voice Cloning Script
Standardizes input audio to 16kHz Mono, generates cloned voice, and logs performance metrics.
"""

import os
import sys
import time
import shutil
import platform
import warnings
import traceback
import psutil

# Suppress warnings
warnings.filterwarnings('ignore')

# --- DEPENDENCY HANDLING ---
try:
    import torch
    import torchaudio
    TORCH_AVAILABLE = True
except (ImportError, OSError) as e:
    TORCH_AVAILABLE = False
    # Will log this in main using the logger

try:
    from pydub import AudioSegment
    from pydub.effects import normalize, compress_dynamic_range
except ImportError:
    # Mock for fallback if pydub missing (should satisfy requirement if installed though)
    pass

# Import structured logging
from logger_config import logger, log_performance

# --- TRANSCRIPTION ---
try:
    import speech_recognition as sr
    TRANSCRIPTION_AVAILABLE = True
except ImportError:
    TRANSCRIPTION_AVAILABLE = False
    # Logged in main


# --- CONFIGURATION ---
INPUT_AUDIO = "../all_inputs/input_audio.wav"
OUTPUT_AUDIO = "../output/output_audio.wav"
# Updated requirement: 16kHz for standardization
TARGET_SAMPLE_RATE = 16000 

def log_system_info():
    """Log system startup Information."""
    logger.info("=== System Startup Info ===")
    logger.info(f"Python Version: {sys.version.split()[0]}")
    logger.info(f"OS/Platform: {platform.platform()}")
    
    if TORCH_AVAILABLE:
        try:
            logger.info(f"Torch Version: {torch.__version__}")
            logger.info(f"CUDA Available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        except Exception as e:
            logger.warning(f"Error checking Torch info: {e}")
    else:
        logger.warning("Torch Version: Not Available (Import Failed)")
        
    logger.info("===========================")

@log_performance
def transcribe_audio(audio_path):
    """
    Transcribe audio content to text using SpeechRecognition (Google API).
    """
    if not TRANSCRIPTION_AVAILABLE:
        logger.warning("SpeechRecognition library not found. Using fallback text.")
        return None

    logger.info(f"Transcribing audio: {audio_path}")
    recognizer = sr.Recognizer()
    
    try:
        with sr.AudioFile(audio_path) as source:
            # record the audio file
            audio_data = recognizer.record(source)
            # transcribe
            text = recognizer.recognize_google(audio_data)
            logger.info(f"Transcription successful: '{text}'")
            return text
            
    except sr.UnknownValueError:
        logger.error("Speech Recognition could not understand audio.")
        return None
    except sr.RequestError as e:
        logger.error(f"Could not request results from Speech Recognition service; {e}")
        return None
    except Exception as e:
        logger.error(f"Error during transcription: {e}")
        return None


@log_performance
def standardize_audio(input_path, output_path):
    """
    Standardize input audio: Mono, 16kHz, Normalized.
    Returns the path to the standardized file.
    """
    logger.info(f"Processing input audio: {input_path}")
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
        
    try:
        audio = AudioSegment.from_file(input_path)
        
        # Convert to Mono
        if audio.channels > 1:
            audio = audio.set_channels(1)
            logger.info("Converted to Mono")
            
        # Resample
        if audio.frame_rate != TARGET_SAMPLE_RATE:
            audio = audio.set_frame_rate(TARGET_SAMPLE_RATE)
            logger.info(f"Resampled to {TARGET_SAMPLE_RATE}Hz")
            
        # Normalize & Compress
        audio = normalize(audio)
        audio = compress_dynamic_range(audio, threshold=-20.0, ratio=2.0)
        
        # Trim silence
        audio = audio.strip_silence(silence_thresh=-40, padding=200)
        
        # Export
        audio.export(output_path, format="wav")
        logger.info(f"Standardized audio saved to: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error standardizing audio: {e}")
        raise RuntimeError(f"Standardization failed and fallback is disabled: {e}")

@log_performance
def generate_audio_qwen3(text, prompt_audio, output_file, **kwargs):
    """
    Generate audio using Qwen3 TTS (or fallback simulation).
    """
    # Cleanup old output to prevent reporting false success on stale files
    if os.path.exists(output_file):
        os.remove(output_file)
        logger.info(f"Existing output file {output_file} removed to ensure fresh synthesis.")

    qwen_model = None
    
    # Try Import Qwen3
    try:
        if TORCH_AVAILABLE:
            from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Log current memory for diagnostic but do NOT block
            available_gb = psutil.virtual_memory().available / (1024**3)
            logger.info(f"System memory available: {available_gb:.2f} GB. Proceeding with model load...")
            
            # Use dtype instead of torch_dtype as per deprecation warning
            qwen_model = Qwen3TTSModel.from_pretrained(
                "Qwen/Qwen3-TTS-12Hz-0.6B-Base", 
                device_map=device,
                dtype=torch.float32
            )
            logger.info("Qwen3 Model Loaded.")
        else:
            raise ImportError("Torch not available. Cannot run Qwen3.")
            
    except Exception as e:
        logger.error(f"Qwen3 Initialization Failed: {e}")
        raise RuntimeError(f"Model initialization failed: {e}")

    logger.info("Execution Mode: STRICT NEURAL SYNTHESIS")

    # Generate 
    if qwen_model:
        try:
            logger.info(f"Synthesizing text: '{text[:30]}...'")
            if hasattr(qwen_model, 'generate_voice_clone'):
                logger.info(f"Generating voice clone (ICL mode) for text: '{text[:50]}...'")
                
                ref_text = kwargs.get('ref_text', None)
                
                try:
                    if ref_text:
                        # Attempt ICL mode (higher quality)
                        audio_list, sr = qwen_model.generate_voice_clone(
                            text=text, 
                            ref_audio=prompt_audio, 
                            ref_text=ref_text,
                            x_vector_only_mode=False
                        )
                    else:
                        raise ValueError("No ref_text provided for ICL mode (In-Context Learning).")
                except Exception as icl_error:
                    logger.warning(f"ICL mode failed: {icl_error}. Trying x_vector_only_mode fallback (Model level)...")
                    # Fallback to x_vector_only_mode (lower quality but works without ref_text)
                    audio_list, sr = qwen_model.generate_voice_clone(
                        text=text, 
                        ref_audio=prompt_audio, 
                        x_vector_only_mode=True
                    )
                
                if audio_list and len(audio_list) > 0:
                    import soundfile as sf
                    sf.write(output_file, audio_list[0], sr)
                    logger.info(f"Generation successful. Saved to {output_file}")
                else:
                    raise ValueError("Qwen3 generated empty audio list.")
            else:
                raise AttributeError("The loaded Qwen3 model does not have 'generate_voice_clone' method.")
                
            logger.info("Qwen3 Generation Complete.")
            return

        except Exception as e:
            logger.error(f"Generation Error: {e}")
            raise RuntimeError(f"Neural synthesis failed: {e}")
    else:
        raise RuntimeError("Qwen3 model was not initialized.")

def main():
    log_system_info()
    logger.info("=== Voice Cloning Pipeline Started ===")
    
    final_text = None
    
    # 1. Standardize Input
    try:
        # Check if input exists, if not use any wav found or warn
        if not os.path.exists(INPUT_AUDIO):
            logger.error(f"No input audio found. Please provide '{INPUT_AUDIO}'.")
            return

        standardized_input = standardize_audio(INPUT_AUDIO, "../output/standardized_input.wav")
    except Exception as e:
        logger.error(f"Critical error in standardization: {e}")
        return

    # 2. Setup Reference Text and Synthesis Text
    ref_text = None
    try:
        # Reference text corresponding to exactly what's in input_audio.wav
        if os.path.exists("../all_inputs/input.txt"):
            with open("../all_inputs/input.txt", "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    ref_text = content
                    logger.info(f"Reference text sourced from input.txt: '{ref_text[:50]}...'")
        else:
            logger.warning("No input.txt found for reference text.")

        # Synthesis text (what the model needs to generate)
        if os.path.exists("../all_inputs/input_1.txt"):
            with open("../all_inputs/input_1.txt", "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    final_text = content
                    logger.info(f"Synthesis text sourced from input_1.txt: '{final_text[:50]}...'")
        else:
            logger.warning("No input_1.txt found. Final text is missing.")
            
        if not final_text:
            logger.error("No text available to synthesize. Exiting.")
            return

    except Exception as e:
        logger.error(f"Error setting up synthesis target text: {e}")

    # 4. Generate Output
    try:
        generate_audio_qwen3(final_text, standardized_input, OUTPUT_AUDIO, ref_text=ref_text)
    except Exception as e:
        logger.error(f"Critical error in generation pipeline: {e}")
        
    logger.info("=== Pipeline Completed ===")

if __name__ == "__main__":
    main()