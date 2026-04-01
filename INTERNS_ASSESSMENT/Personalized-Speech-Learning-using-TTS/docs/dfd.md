# Data Flow Diagrams

This document details the data flow for the core functions of the voice cloning system.

## 1. Preprocess Audio
**Function**: `preprocess_audio_for_best_quality()`

```mermaid
flowchart TB
    subgraph "preprocess_audio_for_best_quality"
        In1([Input Path]) --> Load[Load Audio]
        Load --> Mono[Convert to Mono]
        Mono --> Rate[Resample 22050Hz]
        Rate --> Norm[Normalize Volume]
        Norm --> Comp[Compress Dynamic Range]
        Comp --> Trim[Trim Silence]
        Trim --> Save[Export Temp File]
        Save --> Out1([Output Path])
    end
```

## 2. Clone Voice (Simple)
**Function**: `clone_voice_simple()`

```mermaid
flowchart TB
    subgraph "clone_voice_simple"
        In2([Inputs: Text, Audio, Lang]) --> Pre{Preprocess?}
        Pre --Yes--> DoPre[Call Preprocess]
        Pre --No--> Raw[Use Raw Audio]
        DoPre --> LoadM[Load Model]
        Raw --> LoadM
        LoadM --> Infer[Run Inference]
        Infer --> Save2[Save WAV]
        Save2 --> Out2([Output Path])
    end
```

## 3. Analyze Voice Sample
**Function**: `analyze_voice_sample()`

```mermaid
flowchart TB
    subgraph "analyze_voice_sample"
        In3([Input Path]) --> Load3[Load Audio]
        Load3 --> Calc[Calculate Metrics]
        Calc --> Dur[Duration]
        Calc --> Vol[Loudness]
        Calc --> Sil[Silence Ratio]
        Dur --> Rec[Generate Recommendations]
        Vol --> Rec
        Sil --> Rec
        Rec --> Out3([Analysis Dictionary])
    end
```

## 4. Optimize Voice Sample
**Function**: `optimize_voice_sample()`

```mermaid
flowchart TB
    subgraph "optimize_voice_sample"
        In4([Input Path]) --> Load4[Load Audio]
        Load4 --> Trans[Apply Transformations]
        Trans --> Mono2[Mono]
        Trans --> Rate2[22050Hz]
        Trans --> Trim2[Trim]
        Trim2 --> Save4[Export Optimized]
        Save4 --> Out4([Output Path])
    end
```

## 5. TTS Model Component
**Component**: internal TTS Model Logic

```mermaid
flowchart TB
    subgraph "TTS Model Flow"
        InModel([Inputs: Text, Speaker Latents, Lang]) --> Tok[Tokenizer]
        Tok --> Emb[Text Embeddings]
        Emb --> GPT[GPT Model]
        InModel --Latents--> GPT
        GPT --> Mel[Mel Spectrogram]
        Mel --> Vocoder[HiFiGAN Vocoder]
        Vocoder --> Wave[Waveform Output]
    end
```
