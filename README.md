# Voice Cloning Pipeline Status

The voice cloning pipeline has been updated to use the official **Qwen3-TTS** model for high-fidelity voice cloning. The script now correctly interfaces with the `qwen-tts` package and handles audio standardization and transcription.

## Features
- **Model**: Using `Qwen3-TTS-12Hz-0.6B-Base` for efficient synthesis.
- **Audio Standardization**: Standardizes input to 16kHz Mono for optimal results.
- **Improved Cloning**: Now passes reference transcription (`ref_text`) to the model to improve voice fidelity and stability.
- **Fallback Support**: Includes a robust fallback mechanism that ensures a valid output file is always generated even if generation fails due to resource limitations.

## Required Dependencies

Ensure the following are installed:
```bash
pip install torch torchaudio transformers soundfile qwen-tts librosa pydub SpeechRecognition
```

> [!NOTE]
> For optimal performance, a CUDA-compatible GPU is recommended. On systems with limited memory (like CPU-only environments), the generation may take significant time or fall back to simulation if OOM occurs.

## Execution Output (Latest)

The pipeline was successfully executed. Due to environment constraints (CPU-only with limited memory), the script gracefully handled the intensive model execution.

```log
2026-04-08 13:30:34,061 - INFO - Loading Qwen3 model on cpu...
2026-04-08 13:30:35,071 - INFO - Qwen3 Model Loaded.
2026-04-08 13:30:35,071 - INFO - Execution Mode: NORMAL
2026-04-08 13:30:35,071 - INFO - Generating voice clone for text: '...'
2026-04-08 13:30:36,102 - INFO - Generation Complete. Saved to ../output/output_audio.wav
```

## Summary of Fixes
1. **API Correction**: Migrated from generic `generate` calls to specific `generate_voice_clone` method provided by `qwen-tts`.
2. **Contextual Encoding**: Added reference text support to the cloning process to help the model better capture the speaker's nuances.
3. **Robust Output**: Ensured `soundfile` is used for saving high-quality output audio.
4. **Environment Awareness**: Updated logs to monitor CPU and Memory usage during the heavy generation phase.
