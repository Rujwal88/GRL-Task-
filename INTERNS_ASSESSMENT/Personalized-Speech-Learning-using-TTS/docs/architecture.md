# System Architecture

## High-Level Architecture

The following diagram illustrates the high-level architecture of the Voice Cloning System, fully compatible with Mermaid v8.8.0.

```mermaid
graph TD
    subgraph "User Input Layer"
        Input_Text[Text to Speak]
        Input_Voice[Voice Sample]
    end

    subgraph "Processing Engine"
        APM[Audio Preprocessing]
        VCE[Voice Cloning Engine]
        OG[Output Generation]
    end

    subgraph "External Services"
        HF[Hugging Face Repo]
        FFMPEG[FFmpeg Library]
    end

    %% Component Connections
    Input_Voice --Raw Audio--> APM
    APM --Process--> FFMPEG
    FFMPEG --Result--> APM
  
    Input_Text --Text string--> VCE
    APM --Cleaned WAV--> VCE
  
    HF -.Download Model.-> VCE
  
    VCE --Latents + Text--> OG
    OG --Final WAV--> User((User))

    %% Formatting
    style HF fill:#ffeb3b,stroke:#fbc02d,color:#000
    style FFMPEG fill:#4caf50,stroke:#2e7d32,color:#fff
    style VCE fill:#2196f3,stroke:#1976d2,color:#fff
```

## Component Overview

### 1. User Input Layer
- **Text Input**: The target text to be synthesized.
- **Voice Sample Input**: A reference WAV/MP3 file providing voice characteristics.

### 2. Audio Preprocessing Module
Optimizes the voice sample for the XTTS model.
- **Dependencies**: `pydub`, `FFmpeg`.
- **Operations**: Mono conversion, Resampling (22050Hz), Normalization, Compression, Silence Trimming.

### 3. Voice Cloning Engine (XTTS v2)
The core logic using Coqui TTS.
- **Model Loading**: Fetches `xtts_v2` weights from Hugging Face.
- **Inference**: Computes speaker latents and generates mel-spectrograms.

### 4. Output Generation
- **Synthesis**: Converts spectrograms to waveform.
- **Export**: Saves as `.wav` file.
