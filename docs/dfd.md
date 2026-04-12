# Data Flow Diagrams

This document details the data flow for the core functions of the Qwen3-based voice cloning system.

## 1. Standardize Audio
**Function**: `standardize_audio()`

```mermaid
graph TB
    subgraph "standardize_audio"
        In1(["Raw Audio Path"]) --> Load["Load File"]
        Load --> Mono["Convert to Mono"]
        Mono --> Rate["Resample 16000Hz"]
        Rate --> Norm["Normalize"]
        Norm --> Comp["Compress Range"]
        Comp --> Trim["Trim Silence"]
        Trim --> Export["Export Standardized"]
        Export --> Out1(["standardized_input.wav"])
    end
```

## 2. Generate Audio (Qwen3)
**Function**: `generate_audio_qwen3()`

```mermaid
graph TB
    subgraph "generate_audio_qwen3"
        In2(["Target Text"]) --> ModelLoad["Load Qwen3 Model"]
        RefAudio(["Standardized Audio"]) --> ModelLoad
        RefText(["Anchor Text"]) --> ICL{Has Anchor?}
        
        ICL --Yes--> ICL_Proc["Generate Voice Clone - ICL Mode"]
        ICL --No--> X_Proc["Generate Voice Clone - X-Vector Mode"]
        
        ICL_Proc --> Synth["Neural Synthesis"]
        X_Proc --> Synth
        
        Synth --> OutList["Audio List"]
        OutList --> Write["Soundfile Write"]
        Write --> Final(["output_audio.wav"])
    end
```

## 3. Main Pipeline Flow
**Function**: `main()`

```mermaid
graph LR
    Start(["Start"]) --> Std["Standardize Audio"]
    Std --> SetupAnchor["Setup Anchor Text"]
    SetupAnchor --> SetupTarget["Setup Target Text"]
    SetupTarget --> Gen["Generate Qwen3 Audio"]
    Gen --> End(["End"])

    InputAudio[("input_audio.wav")] --> Std
    InputText[("input.txt")] --> SetupAnchor
    InputText1[("input_1.txt")] --> SetupTarget
```
