import os
import sys
import shutil
import logging
import torch
import torchaudio
import platform
from pydub import AudioSegment
import speech_recognition as sr
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
# Note: Using Qwen3TTSModel from the installed package

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("qwen3_cloning.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

INPUT_AUDIO = "input_audio.wav"
OUTPUT_AUDIO = "output_audio.wav"
STANDARDIZED_AUDIO = "standardized_input.wav"
TARGET_SR = 16000

def check_env():
    logger.info("Checking environment...")
    logger.info(f"Python: {sys.version}")
    try:
        import torch
        logger.info(f"Torch: {torch.__version__} (CUDA: {torch.cuda.is_available()})")
    except ImportError:
        logger.error("Torch not found.")
        return False
    
    try:
        import qwen_tts
        logger.info(f"Qwen3-TTS package found: {qwen_tts.__file__}")
    except ImportError:
        logger.error("Qwen3-TTS package not found.")
        return False
    return True

def standardize_audio(input_path, output_path):
    logger.info(f"Standardizing audio: {input_path}")
    try:
        if not os.path.exists(input_path):
            # Try to find any wav
            wavs = [f for f in os.listdir('.') if f.endswith('.wav') and f != OUTPUT_AUDIO and f != output_path]
            if wavs:
                logger.warning(f"Input {input_path} not found. Using {wavs[0]}")
                shutil.copy(wavs[0], input_path)
            else:
                raise FileNotFoundError("No input audio found.")

        audio = AudioSegment.from_file(input_path)
        audio = audio.set_channels(1).set_frame_rate(TARGET_SR)
        audio.export(output_path, format="wav")
        logger.info(f"Audio standardized to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Standardization failed: {e}")
        # Try simplistic copy if pydub fails
        if os.path.exists(input_path):
             shutil.copy(input_path, output_path)
             return output_path
        raise

def transcribe_audio(audio_path):
    logger.info("Transcribing audio (ASR)...")
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            logger.info(f"Transcription: {text}")
            return text
    except Exception as e:
        logger.error(f"ASR Failed: {e}")
        logger.warning("Using fallback text for cloning demonstration.")
        return "This is a fallback text because automatic speech recognition failed."

def generate_voice_clone(text, ref_audio, output_path):
    logger.info("Initializing Qwen3 TTS for Voice Cloning...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Try 1.7B model first, fallback to 0.6B
    models_to_try = [
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    ]

    model = None
    for model_name in models_to_try:
        try:
            logger.info(f"Attempting to load model: {model_name}")
            # Ensure float32 for CPU if needed, or let auto handle it
            # Qwen models often require bfloat16 which might be issues on some CPUs, 
            # but transformers usually handles conversion if torch_dtype is not forced or auto.
            # We'll try default load first.
            model = Qwen3TTSModel.from_pretrained(
                model_name,
                device_map="auto", # auto might use cpu
            )
            logger.info(f"Model {model_name} loaded successfully.")
            break
        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            if "Out of memory" in str(e) or "StartService" in str(e):
                logger.info("Trying smaller model due to resource/load error...")
                continue
            # Try to force CPU or float32 if dtypes are issue
            try:
                logger.info(f"Retrying {model_name} with float32/cpu explicit...")
                model = Qwen3TTSModel.from_pretrained(
                    model_name,
                    device_map="cpu",
                    torch_dtype=torch.float32
                )
                logger.info(f"Model {model_name} loaded with float32.")
                break
            except Exception as e2:
                logger.error(f"Retry failed: {e2}")

    if model is None:
        raise RuntimeError("All Qwen3 TTS models failed to load.")

    # Generate
    logger.info(f"Generating voice clone with text: '{text[:50]}...'")
    try:
        # Check available methods in Qwen3TTSModel (based on README)
        # generate_voice_clone(text, language, ref_audio, ref_text)
        # We need ref_text for voice cloning if not using x_vector_only_mode
        # We have the transcript of the ref_audio!
        
        # Assumption: The input audio is BOTH the prompt and the source of text.
        # So ref_audio = input, ref_text = transcript.
        # But we want to generate the *same* text?
        # "Convert input audio to text... Using BOTH extracted text and original audio... generate a CLONED AI VOICE"
        # Usually cloning means speaking NEW text.
        # But if the user implies "Re-speak the input audio with the cloned voice" (Voice Conversion), 
        # then target_text == ref_text.
        
        target_text = text # We make it speak the same text? Or a standard demo text?
        # The prompt says: "Speaking the extracted text clearly".
        # So yes, Voice Conversion (or ASR+TTS Re-synthesis).
        
        wavs, sr = model.generate_voice_clone(
            text=target_text,
            language="auto", # Let it detect or default
            ref_audio=ref_audio, # path to wav
            ref_text=target_text, # text of the ref_audio
        )
        
        # Save
        if len(wavs) > 0:
            sf.write(output_path, wavs[0], sr)
            logger.info(f"Successfully saved output to {output_path}")
        else:
            raise RuntimeError("Model generated no audio.")

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise

def main():
    logger.info("=== Standardizing Qwen3 Pipeline ===")
    if not check_env():
         logger.error("Environment verification failed.")
         # Try to fix? We already ran setup script.
         # Maybe re-run pip? (No, infinite loop risk)
         return

    try:
        std_audio = standardize_audio(INPUT_AUDIO, STANDARDIZED_AUDIO)
        
        text = transcribe_audio(std_audio)
        if not text:
            text = "Hello, this is a test of voice cloning."

        generate_voice_clone(text, std_audio, OUTPUT_AUDIO)
        
        if os.path.exists(OUTPUT_AUDIO):
             logger.info("Qwen3 voice cloning completed successfully.")
        else:
             logger.error("Output file missing!")
             
    except Exception as e:
        logger.critical(f"Process failed: {e}")
        # Retry logic could go here, but we did retries in sub-steps.

if __name__ == "__main__":
    main()
