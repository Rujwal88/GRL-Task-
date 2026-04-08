
import torch
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test():
    try:
        device = "cpu"
        model_id = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
        logger.info(f"Loading {model_id} on {device}...")
        # Try loading with low_cpu_mem_usage if applicable
        model = Qwen3TTSModel.from_pretrained(model_id, device_map=device)
        logger.info("Success!")
    except Exception as e:
        logger.error(f"Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test()
