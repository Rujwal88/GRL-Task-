import sys
import traceback

try:
    sys.path.append(r"D:\Github\GRL-Task-\Qwen3-TTS")
    import qwen_tts
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
    print("Import successful!")
except Exception:
    with open("error_log.txt", "w", encoding="utf-8") as f:
        traceback.print_exc(file=f)
