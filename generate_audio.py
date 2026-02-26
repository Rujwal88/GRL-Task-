import os
import sys
import torch
import soundfile as sf
import logging
import numpy as np
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("execution.log")
    ]
)
logger = logging.getLogger(__name__)

def install_dependencies():
    logger.info("Step 2: Checking and installing dependencies...")
    deps = ["torch", "transformers", "accelerate", "soundfile", "librosa", "sentencepiece"]
    for dep in deps:
        try:
            __import__(dep)
            logger.info(f"Dependency {dep} is already present.")
        except ImportError:
            logger.info(f"Installing missing dependency: {dep}")
            try:
                if dep == "torch":
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cpu"])
                else:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            except Exception as e:
                logger.error(f"Failed to install {dep}: {e}")

def run_tts():
    try:
        # Step 5: Read input.txt
        input_path = "input.txt"
        if not os.path.exists(input_path):
            logger.error("input.txt NOT FOUND!")
            return False
            
        with open(input_path, "r", encoding="utf-8") as f:
            full_text = f.read().strip()
            
        if not full_text:
            logger.error("input.txt IS EMPTY!")
            return False

        # Step 3 & 4: Load model
        # We try the 0.6B model first because 7B is impossible on 4GB RAM and 0.6B is cached.
        # User requested Qwen3TTS (Qwen Audio).
        models_to_try = [
            "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            "Qwen/Qwen2-Audio-1.8B",
            "Qwen/Qwen2-Audio-7B-Instruct" # Last resort, unlikely to work
        ]
        
        # Check if we can use the specialized Qwen3-TTS package logic
        try:
            from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
            logger.info("Using specialized Qwen3TTSModel interface.")
            
            model_id = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
            logger.info(f"Loading {model_id} on CPU...")
            model = Qwen3TTSModel.from_pretrained(
                model_id,
                device_map="cpu",
                torch_dtype=torch.float32
            )
            
            # Qwen3-TTS requires a reference audio for cloning
            ref_audio = "standardized_ref.wav"
            if not os.path.exists(ref_audio):
                # Try to find any other wav file to use as reference
                wavs = [f for f in os.listdir('.') if f.endswith('.wav') and f != "output_audio.wav"]
                if wavs:
                    ref_audio = wavs[0]
                    logger.info(f"Using {ref_audio} as reference audio.")
                else:
                    logger.error("No reference audio found for voice cloning!")
                    return False
            
            logger.info("Creating voice clone prompt...")
            # We'll use x_vector_only_mode if we don't have the text for the ref audio
            # But the model usually works better with ref_audio
            prompt_items = model.create_voice_clone_prompt(
                ref_audio=ref_audio,
                x_vector_only_mode=True
            )
            
            # Step 6: Convert text to speech
            # Split text to avoid memory spikes
            chunks = [full_text[i:i+400] for i in range(0, len(full_text), 400)]
            all_wavs = []
            final_sr = 24000
            
            for i, chunk in enumerate(chunks):
                logger.info(f"Generating chunk {i+1}/{len(chunks)}...")
                wavs, sr_out = model.generate_voice_clone(
                    text=chunk, 
                    language="auto",
                    voice_clone_prompt=prompt_items
                )
                if wavs and len(wavs) > 0:
                    all_wavs.append(wavs[0])
                    final_sr = sr_out
            
            if all_wavs:
                final_wav = np.concatenate(all_wavs)
                sf.write("output_audio.wav", final_wav, final_sr)
                logger.info("output_audio.wav generated successfully.")
                return True
                
        except Exception as e:
            import traceback
            logger.warning(f"Specialized Qwen3-TTS failed: {e}")
            logger.debug(traceback.format_exc())
            # logger.warning("Trying generic transformers logic...") # We'll stop here if ONLY Qwen3TTS is allowed

        return False


    except Exception as e:
        logger.critical(f"Global error in run_tts: {e}")
        return False

def verify_output():
    # Step 8: Verify file exists and is non-empty
    if not os.path.exists("output_audio.wav"):
        logger.error("Verification failed: output_audio.wav does not exist.")
        return False
        
    size = os.path.getsize("output_audio.wav")
    if size < 1000:
        logger.error(f"Verification failed: output_audio.wav is too small ({size} bytes).")
        return False
        
    try:
        import librosa
        y, sr = librosa.load("output_audio.wav")
        duration = librosa.get_duration(y=y, sr=sr)
        logger.info(f"Generated audio duration: {duration:.2f} seconds")
        if duration > 1.0:
            return True
        else:
            logger.error("Verification failed: Audio duration is less than 1 second.")
            return False
    except Exception as e:
        logger.error(f"Verification failed during duration check: {e}")
        return False

if __name__ == "__main__":
    install_dependencies()
    success = run_tts()
    
    if success and verify_output():
        logger.info("FINAL CONDITION MET: Audio generated and verified.")
        sys.exit(0)
    else:
        logger.error("FINAL CONDITION NOT MET.")
        sys.exit(1)
