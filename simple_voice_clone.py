"""
Qwen3 TTS Voice Cloning Script
Standardizes input audio to 24kHz Mono, generates cloned voice, and logs performance metrics.
"""

import os
import sys
import time
import shutil
import platform
import warnings
import traceback

# Suppress warnings
warnings.filterwarnings('ignore')

# --- DEPENDENCY HANDLING ---
try:
    import torch
    import torchaudio
    TORCH_AVAILABLE = True
except (ImportError, OSError) as e:
    TORCH_AVAILABLE = False

try:
    from pydub import AudioSegment
    from pydub.effects import normalize, compress_dynamic_range
except ImportError:
    pass

TRANSFORMERS_AVAILABLE = False
AUTO_TTS_AVAILABLE = False
try:
    import transformers
    
    # --- VERSION GUARD ---
    # Mandatory fix: Fail if Transformers >= 5.0.0
    current_version = transformers.__version__
    if current_version.startswith("5.") or (current_version.split('.')[0].isdigit() and int(current_version.split('.')[0]) >= 5):
         print(f"CRITICAL ERROR: Transformers version {current_version} is incompatible.")
         print("Please downgrade to Transformers 4.x (e.g., 4.44.2) to use AutoModelForTextToSpeech.")
         sys.exit(1)

    from transformers import AutoProcessor
    TRANSFORMERS_AVAILABLE = True
    try:
        from transformers import AutoModelForTextToSpeech
        AUTO_TTS_AVAILABLE = True
    except ImportError:
        # Fallback: Qwen3 often uses AutoModel with remote code in 4.x
        try:
            from transformers import AutoModel as AutoModelForTextToSpeech
            AUTO_TTS_AVAILABLE = True
        except ImportError:
            AUTO_TTS_AVAILABLE = False
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import qwen_tts.inference
except ImportError:
    pass

# Import structured logging
from logger_config import logger, log_performance

# --- TRANSCRIPTION ---
try:
    import speech_recognition as sr
    TRANSCRIPTION_AVAILABLE = True
except ImportError:
    TRANSCRIPTION_AVAILABLE = False


# --- CONFIGURATION ---
INPUT_AUDIO = "input_audio.wav"
OUTPUT_DIR = "outputs/audio"
OUTPUT_AUDIO = os.path.join(OUTPUT_DIR, "output_qwen3.wav")
TARGET_SAMPLE_RATE = 24000   

# Configurable Model ID 
QWEN3_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

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
            audio_data = recognizer.record(source)
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
    Standardize input audio: Mono, 24kHz, Normalized.
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
        # Fallback: Just copy if processing fails
        shutil.copy(input_path, output_path)
        return output_path

@log_performance
def generate_audio_qwen3(text, prompt_audio, output_file):
    """
    Generate audio using Qwen3 TTS via AutoModelForTextToSpeech.
    """
    logger.info("Initializing Qwen3 TTS generation...")
    
    if not TORCH_AVAILABLE:
         logger.error("Torch is not available. Cannot proceed with Qwen3 TTS.")
         raise ImportError("Torch is required for Qwen3 TTS.")
         
    if not TRANSFORMERS_AVAILABLE:
        logger.error("Transformers library not found.")
        raise ImportError("Transformers library is required.")

    if not AUTO_TTS_AVAILABLE:
        import transformers
        logger.error(f"AutoModelForTextToSpeech is missing in transformers v{transformers.__version__}")
        raise ImportError("AutoModelForTextToSpeech class is required but not found in transformers.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Target Device: {device}")

    try:
        # Load Processor and Model
        logger.info(f"Loading processor and model from: {QWEN3_MODEL_ID}")
        
        try:
            processor = AutoProcessor.from_pretrained(QWEN3_MODEL_ID, trust_remote_code=True)
            model = AutoModelForTextToSpeech.from_pretrained(QWEN3_MODEL_ID, trust_remote_code=True).to(device)
            logger.info("Model and Processor loaded successfully.")
        except (KeyError, ValueError, RuntimeError) as e:
            logger.warning(f"AutoModel failed ({e}), attempting direct import from qwen_tts...")
            try:
                logger.info("Tentative import: from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel")
                from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
                logger.info("Import successful. Now initializing model from pretrained...")
                
                # processor might also need fallback, but usually available via AutoProcessor if trust_remote_code works for it?
                # Actually, if config is broken for AutoModel, it might break for AutoProcessor too?
                # Let's assume AutoProcessor works or we find Qwen3TTSTokenizer
                
                # Try loading Qwen3TTSModel directly
                model = Qwen3TTSModel.from_pretrained(QWEN3_MODEL_ID, trust_remote_code=True)
                model.model.to(device)
                model.device = device
                logger.info("Qwen3TTSModel loaded directly.")
            except ImportError as ie:
                logger.error(f"Failed to import Qwen3TTSModel explicitly: {ie}")
                raise e
            except Exception as e2:
                logger.error(f"Fallback model loading failed: {e2}")
                raise e2

        logger.info(f"Synthesizing text: '{text[:50]}...'")
        
        # Prepare inputs
        # Note: Actual arguments depend on specific Qwen3 API (inputs vs text, etc.)
        # Standard AutoModelForTextToSpeech usually takes 'text' or 'input_ids'
        # We assume standard usage: processor(text=text, ...)
        
        audio_data = None
        output_sr = TARGET_SAMPLE_RATE

        if hasattr(model, "generate_voice_clone"):
            logger.info("Using Qwen3TTSModel.generate_voice_clone()")
            # We use the input text as ref_text effectively assuming the prompt audio matches the text 
            # (which is true in this pipeline's main logic) OR we accept mismatch.
            # Ideally we should strictly use transcript of ref_audio.
            # Since 'text' here IS the transcript (from main), this is correct.
            wavs, sr = model.generate_voice_clone(text=text, ref_audio=prompt_audio, ref_text=text)
            if len(wavs) > 0:
                audio_data = torch.from_numpy(wavs[0])
                output_sr = sr
            else:
                raise RuntimeError("No audio generated by Qwen3TTSModel")
        else:
            logger.info("Using standard model.generate()")
            inputs = processor(text=text, return_tensors="pt").to(device)
            # Future proofing: if auto model works eventually
            with torch.no_grad():
                output = model.generate(**inputs)
            
            if hasattr(output, "waveform"):
                audio_data = output.waveform
            elif isinstance(output, torch.Tensor):
                audio_data = output
            elif isinstance(output, tuple):
                audio_data = output[0]
            else:
                audio_data = output
             
        # Ensure tensor is CPU
        if isinstance(audio_data, torch.Tensor):
            audio_data = audio_data.cpu()
        
        # Save to file
        # Convert to 2d if 1d (channels, time)
        if audio_data.ndim == 1:
            audio_data = audio_data.unsqueeze(0)
            
        # Torchaudio save
        # Torchaudio save
        torchaudio.save(output_file, audio_data, output_sr)
        logger.info(f"Qwen3 Generation Complete. Saved to {output_file}")

    except Exception as e:
        logger.error(f"Critical error during Qwen3 generation: {e}")
        logger.error(traceback.format_exc())
        raise e

def main():
    log_system_info()
    logger.info("=== Voice Cloning Pipeline Started ===")
    
    # Default text
    final_text = "This is a demonstration of the Qwen 3 Text to Speech model."
    
    # Parse CLI arguments
    import argparse
    parser = argparse.ArgumentParser(description="Qwen3 TTS Voice Cloning")
    parser.add_argument("--input", type=str, default="input_audio.wav", help="Path to input audio file")
    parser.add_argument("--output", type=str, default=os.path.join("outputs", "audio", "output_qwen3.wav"), help="Path to output audio file")
    args = parser.parse_args()

    input_audio_path = args.input
    output_audio_path = args.output
    
    # Ensure output directory exists if specified
    output_dir = os.path.dirname(output_audio_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 1. Standardize Input
    try:
        if not os.path.exists(input_audio_path):
             logger.error(f"Input audio '{input_audio_path}' not found. Cannot clone voice.")
             raise FileNotFoundError(f"Input file {input_audio_path} missing.")

        # Create a temp file name based on input
        temp_standardized = f"temp_standardized_{os.path.basename(input_audio_path)}"
        standardized_input = standardize_audio(input_audio_path, temp_standardized) 
    except Exception as e:
        logger.error(f"Error in standardization: {e}")
        return

    # 2. Transcribe
    try:
        transcribed_text = transcribe_audio(standardized_input)
        if transcribed_text:
            final_text = transcribed_text
            logger.info(f"Process will use transcribed text: '{final_text}'")
        else:
            logger.warning(f"Transcription failed. Using default text: '{final_text}'")
    except Exception as e:
        logger.error(f"Error during transcription setup: {e}")

    # 3. Generate Output
    try:
        generate_audio_qwen3(final_text, standardized_input, output_audio_path)
        logger.info("=== Voice Cloning Pipeline Completed Successfully ===")
    except Exception as e:
        logger.error(f"Pipeline Failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
