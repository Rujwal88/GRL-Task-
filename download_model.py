import os
import sys

# Enable HF Transfer for speed
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

try:
    from huggingface_hub import snapshot_download
    
    model_id = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    print(f"Starting optimized download for {model_id}...")
    
    # Download everything explicitly
    snapshot_download(
        repo_id=model_id, 
        allow_patterns=["*.json", "*.safetensors", "*.py"],
        resume_download=True
    )
    print("Download confirmed complete.")
    
except Exception as e:
    print(f"Download failed: {e}")
    sys.exit(1)
