# Logging System Guide

This project implements a robust logging system compliant with enterprise standards.

## 1. Configuration (`logger_config.py`)

- **Log File**: `logs/voice_cloning.log`
- **RotationPolicy**: 5MB max size, 5 backups.
- **Level**: `DEBUG` (capture all details).
- **Format**: `Timestamp - Level - Module - Function - Message`.

## 2. Features

### Performance and Complexity Tracking
The `@log_performance` decorator automatically captures:
- **Execution Time**: In seconds.
- **Memory Usage**: RSS Delta (MB).
- **Return Value**: Type and value (truncated).
- **Input Args**: Logged at DEBUG level on entry.

### Logic-Specific Logging
- **Complexity Notes**: Key algorithms (O(N)) are noted in logs or comments.
- **Object Data**: Text lengths, audio durations, and file sizes are logged.
- **System Info**: Python, PyTorch, and Hardware status logged at startup.

## 3. How to Read Logs

**Example Entry:**
```text
2024-01-23 21:00:00,123 - INFO - simple_voice_clone - preprocess_audio_for_best_quality() - 🔧 Preprocessing audio...
2024-01-23 21:00:00,500 - DEBUG - simple_voice_clone - preprocess_audio_for_best_quality() - [Time] Mono Conversion: 0.005s
2024-01-23 21:00:01,000 - INFO - simple_voice_clone - wrapper() - Exiting preprocess_audio_for_best_quality - Duration: 0.88s - Return: str = "temp.wav"
```

## 4. Error Handling
Exceptions are caught and logged with `CRITICAL` level including full stack traces (`exc_info=True`).
