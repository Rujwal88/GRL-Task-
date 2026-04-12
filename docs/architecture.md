# System Architecture

## High-Level Architecture

The following diagram illustrates the high-level architecture of the Voice Cloning System, using Mermaid v8.8.0.

```mermaid
graph TD
    subgraph "User Input Layer"
        Input_Text["Synthesis Target Text<br/>(input_1.txt / input.txt)"]
        Input_Anchor["Anchor Text<br/>(input.txt / Transcription)"]
        Input_Voice["Reference Audio<br/>(input_audio.wav)"]
    end

    subgraph "Processing Engine"
        SA["Standardize Audio<br/>(16kHz Mono)"]
        TA["Transcribe Audio<br/>(Fallback Anchor)"]
        Q3["Qwen3 TTS Engine"]
        OG[Output Generation]
    end

    subgraph "External Dependencies"
        Q3M["Qwen3 Model Weights"]
        PD["Pydub / FFmpeg"]
        SR["SpeechRecognition API"]
    end

    %% Flow
    Input_Voice --> SA
    SA --> PD
    PD --> SA
    
    Input_Anchor --> TA
    TA --> SR
    SR --> TA
    
    Input_Text --> Q3
    SA --Standardized WAV--> Q3
    TA --Anchor Text--> Q3
    Q3M --Model Weights--> Q3
    
    Q3 --Audio List--> OG
    OG --Final WAV--> User((User))

    %% Styles
    style Q3M fill:#fbc02d,stroke:#f57f17,color:#000
    style PD fill:#4caf50,stroke:#2e7d32,color:#fff
    style Q3 fill:#2196f3,stroke:#1976d2,color:#fff
```

## Component Overview

### 1. User Input Layer
- **Synthesis Target**: The text to be spoken. Sourced from `input_1.txt` (priority) or `input.txt` (fallback).
- **Anchor Text**: Describes the content of the reference audio. Used for In-Context Learning (ICL) to improve cloning quality. Sourced from `input.txt` or automatic transcription.
- **Reference Audio**: High-quality voice sample providing the target timbre and style.

### 2. Audio Standardization Module (`standardize_audio`)
Optimizes the voice sample for the Qwen3 model.
- **Specifications**: Mono, **16,000Hz** sample rate.
- **Operations**: Normalization, Dynamic Range Compression, and Silence Trimming.

### 3. Voice Cloning Engine (Qwen3 TTS)
Strict neural synthesis using the `Qwen/Qwen3-TTS-12Hz-0.6B-Base` model.
- **ICL Mode**: Uses the anchor text and reference audio for high-fidelity zero-shot cloning.
- **X-Vector Fallback**: If anchor text is unavailable, the model falls back to embedding-only synthesis.

### 4. Output Generation
- **Synthesis**: Generates audio lists via the Qwen3 inference pipeline.
- **Export**: Saves final standardized output to `../output/output_audio.wav` using `soundfile`.
