import sys

print("\n--- TORCH ---")
try:
    import torch
    print(f"Torch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"Torch Import Failed: {e}")
except OSError as e:
    print(f"Torch OSError: {e}")

print("\n--- TRANSFORMERS ---")
try:
    import transformers
    print(f"Transformers Version: {transformers.__version__}")
    from transformers import AutoProcessor
    print("AutoProcessor imported successfully.")
except ImportError as e:
    print(f"Transformers Import Failed: {e}")
except Exception as e:
    print(f"Transformers Error: {e}")

print("\n--- QWEN ---")
try:
    import qwen_tts
    print("qwen_tts imported.")
except ImportError as e:
    print(f"Qwen Import Failed: {e}")
