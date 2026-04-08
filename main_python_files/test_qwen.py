import torch
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
import soundfile as sf
import os

model_name = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
device = "cpu"

print(f"Loading model {model_name}...")
model = Qwen3TTSModel.from_pretrained(
    model_name,
    device_map=device,
    torch_dtype=torch.float32
)
print("Model loaded.")

text = "Hello, this is a test of voice cloning."
prompt_audio = "../all_inputs/input_audio.wav"
output_file = "test_output.wav"

print(f"Generating voice clone for text: {text}")
try:
    audio_list, sr = model.generate_voice_clone(text=text, ref_audio=prompt_audio)
    if audio_list:
        sf.write(output_file, audio_list[0], sr)
        print(f"Success! Saved to {output_file}")
    else:
        print("Error: Empty audio list returned.")
except Exception as e:
    print(f"Generation failed: {e}")
    import traceback
    traceback.print_exc()
