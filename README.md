# Qwen3-TTS Voice Cloning Pipeline

This project implements a high-fidelity voice cloning pipeline using the **Qwen3-TTS** model. It automates the process of standardizing reference audio, transcribing content, and synthesizing new speech with a cloned voice.

## 🛠 Requirements

To run this pipeline, ensure your system meets the following software requirements:

### Core Dependencies
- **Python**: 3.10 or higher
- **FFmpeg**: Required for audio file processing (pydub)
- **Git**: Required for cloning the Qwen3-TTS repository

### Python Packages
Install the necessary libraries using `pip`:
```bash
pip install torch torchaudio transformers soundfile librosa pydub SpeechRecognition accelerate
```

### Model Access
The pipeline uses the `Qwen/Qwen3-TTS-12Hz-0.6B-Base` model. The `qwen-tts` package or a manual clone of the [Qwen3-TTS repository](https://github.com/QwenLM/Qwen3-TTS) is required.

> [!TIP]
> A CUDA-compatible GPU is highly recommended for real-time synthesis. Systems relying on CPU will experience significantly slower generation times.

---

## 🚀 How to Run

Follow these steps to execute the voice cloning pipeline:

### 1. Environment Setup
Run the provided PowerShell script to configure your environment and install the specialized Qwen3-TTS package:
```powershell
./setup_env.ps1
```

### 2. Prepare Inputs
Place your reference audio in the `all_inputs/` directory:
- Reference Audio: `all_inputs/input_audio.wav`
- (Optional) Target Text: `all_inputs/input.txt` (If omitted, the script will transcribe the reference audio).

### 3. Execute the Pipeline
Run the specific execution file from the root or from within its directory:
```bash
# Recommended: Run from within the python files directory to ensure relative paths resolve correctly
cd main_python_files
python simple_voice_clone.py
```

---

## 📂 Core Components

| File Path | Description |
| :--- | :--- |
| `main_python_files/simple_voice_clone.py` | **The Main Execution File.** Handles the entire orchestration: pre-processing, transcription, and synthesis. |
| `main_python_files/logger_config.py` | Provides structured logging for monitoring performance and errors. |
| `setup_env.ps1` | Automation script for environment preparation and dependency installation. |
| `all_inputs/` | Source directory for reference audio and target text. |
| `output/` | Destination for all generated artifacts. |

---

## 🔄 Project Process & Output Generation

The voice cloning pipeline follows a structured five-step process:

1. **Environment Initialization**: The system checks for available hardware (CPU/GPU) and verifies that all core dependencies (`torch`, `torchaudio`, etc.) are correctly installed and loaded.
2. **Audio Standardization**: 
   - The reference audio (`all_inputs/input_audio.wav`) is loaded.
   - It is converted to **Mono** and resampled to **16kHz**.
   - The script applies normalization, dynamic range compression, and trims silence to ensure the reference signal is clean and optimized for the model.
   - The result is saved as `output/standardized_input.wav`.
3. **Transcription & Text Setup**:
   - The system uses the Google Speech Recognition API to transcribe the reference audio into text.
   - If a target text file exists at `all_inputs/input.txt`, that text is used for synthesis. Otherwise, the transcribed text itself is used as the target text.
4. **Voice Cloning (Synthesis)**:
   - The `Qwen3TTSModel` is loaded with its pre-trained weights.
   - The model uses the "zero-shot" voice cloning capability to synthesize the target text using the voice characteristics of the standardized reference audio.
   - The output is generated and saved using `soundfile` to ensure high fidelity.
5. **Execution Logging**: 
   - Every step, including performance metrics and resource usage (CPU/Memory), is recorded in the structured logs located in `main_python_files/logs/`.

### How Output is Generated (Technical Detail)
The output `output_audio.wav` is generated through a **two-stream synthesis process**:
1. **Acoustic Extraction**: The model extracts the unique frequency and timbral features from the standardized reference audio.
2. **Neural Synthesis**: The target text is processed alongside the reference features to generate a high-fidelity waveform. The result is then written to disk using the `soundfile` library to maintain bit-depth and sample-rate integrity.

---

## 📤 Project Output

Upon successful execution, the following outputs are generated in the `output/` directory:

| Artifact | Description |
| :--- | :--- |
| `standardized_input.wav` | The refined and pre-processed version of your reference voice. |
| `output_audio.wav` | **The Final Output:** The synthesized speech with the cloned voice. |
| `logs/` | Detailed execution history and performance benchmarks. |

### Latest Execution Log (Summary)
```log
2026-04-08 13:30:34,061 - INFO - Loading Qwen3 model on cpu...
2026-04-08 13:30:35,071 - INFO - Qwen3 Model Loaded.
2026-04-08 13:30:35,071 - INFO - Execution Mode: NORMAL
2026-04-08 13:30:35,102 - INFO - Generation Complete. Saved to ../output/output_audio.wav
```

---

## 🔗 Verification

You can verify the latest outputs and project progress at the following link:

**[GitHub Project Repository](https://github.com/Rujwal88/GRL-Task-)**

> [!NOTE]
> The `output` folder in the repository contains the most recent generation results and standardized audio files for review.
