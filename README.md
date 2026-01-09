# Text-to-Speech Voice Cloning

A high-quality voice cloning system using Coqui TTS (XTTS v2) that allows you to clone any voice and generate natural-sounding speech in multiple languages. This project includes tools for voice sample analysis and optimization to achieve the best possible cloning results.

## 🎯 Features

- **High-Quality Voice Cloning**: Uses XTTS v2 model for maximum accuracy and naturalness
- **Multilingual Support**: Supports 17+ languages including English, Spanish, French, German, Italian, Portuguese, Polish, Turkish, Russian, Dutch, Czech, Arabic, Chinese, Japanese, Korean, and Hungarian
- **Automatic Audio Preprocessing**: Optimizes voice samples automatically for best results
- **Voice Sample Analyzer**: Analyze and improve your voice samples before cloning
- **GPU Acceleration**: Automatically uses GPU if available for faster processing
- **Quality Optimization**: Pre-configured settings for maximum voice similarity

## 📋 Requirements

### System Requirements
- Python 3.8 or higher
- **Internet connection** (required for first-time model download)
- CUDA-capable GPU (optional, but recommended for faster processing)
- FFmpeg (required for audio processing)
- **~2GB free disk space** (for model storage)

### Python Dependencies
All dependencies are listed in `requirements.txt`:
- `TTS>=0.22.0` - Core text-to-speech library
- `torch>=2.0.0` - Deep learning framework
- `pydub>=0.25.1` - Audio processing library

Additional dependencies (automatically installed with TTS):
- numpy
- scipy
- librosa
- transformers
- encodec

## 🚀 Installation

### 1. Clone or Download the Project
```bash
cd /path/to/text-to-speech
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Install FFmpeg (Required for Audio Processing)

**macOS:**
```bash
brew install ffmpeg
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

**Windows:**
Download from [FFmpeg website](https://ffmpeg.org/download.html) or use:
```bash
choco install ffmpeg
```

### 4. Verify Installation
```bash
python -c "from TTS.api import TTS; print('TTS installed successfully!')"
```

### 5. First-Time Model Download

**Important:** On first run, the XTTS v2 model will be automatically downloaded from Hugging Face. This requires:
- **Active internet connection**
- **~1.5GB download** (model size)
- **Storage location**: Models are stored in `~/.local/share/tts/` (Linux/macOS) or `%USERPROFILE%\.local\share\tts\` (Windows)

The model name used is: `tts_models/multilingual/multi-dataset/xtts_v2`

**Note:** No API keys or authentication required - the model is downloaded directly from Hugging Face's public model repository.

## 📁 Project Structure

```
text-to-speech/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── simple_voice_clone.py        # Main voice cloning script
├── improve_voice_sample.py      # Voice sample analyzer and optimizer
├── myvoice.wav                  # Your voice sample (example)
├── myvoice_optimized.wav        # Optimized voice sample (generated)
├── output_max_quality.wav       # Generated cloned voice output (default)
└── temp_preprocessed_voice.wav  # Temporary file (auto-deleted after processing)
```

## 🎙️ Usage

### Quick Start: Voice Cloning

1. **Prepare Your Voice Sample**
   - Record at least 6-10 seconds of clear speech (15-20 seconds recommended)
   - Save as WAV format (e.g., `myvoice.wav`)
   - Place in the project directory

2. **Edit the Script**
   Open `simple_voice_clone.py` and modify the configuration section:
   ```python
   TEXT_TO_SPEAK = "Your text here..."
   VOICE_SAMPLE = "myvoice.wav"
   LANGUAGE = "en"  # Language code
   ```

3. **Run the Script**
   ```bash
   python simple_voice_clone.py
   ```

4. **Find Your Output**
   The cloned voice will be saved as `output_max_quality.wav`

### Advanced: Optimize Your Voice Sample First

For best results, analyze and optimize your voice sample before cloning:

```bash
python improve_voice_sample.py
```

This script will:
1. Analyze your voice sample and provide quality recommendations
2. Create an optimized version (`myvoice_optimized.wav`)
3. Show a comparison between original and optimized samples

Then use the optimized file in `simple_voice_clone.py`:
```python
VOICE_SAMPLE = "myvoice_optimized.wav"
```

### Programmatic Usage

You can also use the functions in your own scripts:

```python
from simple_voice_clone import clone_voice_simple

# Clone voice
output_file = clone_voice_simple(
    text="Hello, this is a test of voice cloning.",
    speaker_audio="myvoice.wav",
    output_file="output.wav",
    language="en",
    preprocess=True  # Auto-optimize audio
)
```

```python
from improve_voice_sample import analyze_voice_sample, optimize_voice_sample

# Analyze voice sample
results = analyze_voice_sample("myvoice.wav")

# Optimize voice sample
optimized = optimize_voice_sample("myvoice.wav", "optimized.wav")
```

## 🌍 Supported Languages

The following language codes are supported:
- `en` - English
- `es` - Spanish
- `fr` - French
- `de` - German
- `it` - Italian
- `pt` - Portuguese
- `pl` - Polish
- `tr` - Turkish
- `ru` - Russian
- `nl` - Dutch
- `cs` - Czech
- `ar` - Arabic
- `zh-cn` - Chinese (Simplified)
- `ja` - Japanese
- `ko` - Korean
- `hu` - Hungarian

## ⚙️ Configuration Options

### Model Configuration

The script uses the following model configuration:
- **Model**: `tts_models/multilingual/multi-dataset/xtts_v2`
- **Source**: Hugging Face Model Hub (public repository)
- **No API keys required**: Model is publicly available
- **GPU detection**: Automatically enabled if CUDA is available

### Voice Cloning Parameters

In `simple_voice_clone.py`, you can adjust these parameters in the `tts.tts_to_file()` call:

- `temperature` (default: 0.05) - Lower values = more similar to reference voice
- `repetition_penalty` (default: 10.0) - Prevents repetitive speech
- `speed` (default: 0.98) - Speech speed (0.5-2.0)
- `length_penalty` (default: 1.0) - Controls sentence length
- `enable_text_splitting` (default: True) - Better handling of long texts

### Audio Preprocessing

The preprocessing automatically:
- Converts stereo to mono
- Sets sample rate to 22050 Hz (optimal for XTTS)
- Normalizes volume
- Applies dynamic range compression
- Trims silence from edges
- Limits to 30 seconds (uses middle section if longer)

## 💡 Tips for Best Results

### Recording Your Voice Sample

1. **Duration**: Record 15-20 seconds of clear speech (minimum 6 seconds)
2. **Environment**: Record in a quiet room with minimal background noise
3. **Microphone**: Use a good quality microphone if possible
4. **Speaking Style**: Speak naturally, as if having a conversation (not reading)
5. **Content**: Use varied sentences with different emotions and intonations
6. **Format**: Save as WAV format for best quality

### What to Avoid

- ❌ Background music or noise
- ❌ Multiple speakers in the recording
- ❌ Very short samples (< 6 seconds)
- ❌ Overly long samples (> 30 seconds will be trimmed)
- ❌ Low-quality recordings or heavy compression

### Improving Results

1. **Use the analyzer**: Run `improve_voice_sample.py` first
2. **Use optimized sample**: Always use the optimized version for cloning
3. **Record multiple samples**: Try different recordings and compare results
4. **Adjust parameters**: Experiment with temperature and speed settings
5. **Check quality score**: Aim for 80%+ quality score from the analyzer

## 🔧 Troubleshooting

### Common Issues

**Error: "Voice sample not found"**
- Make sure the file path is correct
- Check that the file exists in the project directory

**Error: "No module named 'pydub'"**
- Install dependencies: `pip install -r requirements.txt`

**Error: "ffmpeg not found"**
- Install FFmpeg (see Installation section)
- On macOS: `brew install ffmpeg`

**Poor quality results**
- Use the voice sample analyzer to check quality
- Record a new sample following the tips above
- Try the optimized version of your sample
- Ensure your sample is at least 10 seconds long

**Slow processing / First run takes long time**
- **First run**: Model downloads from Hugging Face (~1.5GB download, requires internet)
- Model is cached locally after first download (no internet needed for subsequent runs)
- Use GPU if available (automatically detected)
- Processing time depends on text length and hardware
- Model location: `~/.local/share/tts/` (can be deleted to force re-download)

**Internet connection required on first run**
- The XTTS v2 model is downloaded from Hugging Face on first use
- No API keys or authentication needed
- After first download, works offline
- If download fails, check internet connection and try again

**Out of memory errors**
- Close other applications
- Use a shorter voice sample
- Process shorter text segments
- If using GPU, ensure sufficient VRAM (recommended: 4GB+)

**PyTorch serialization warnings**
- The script includes a workaround for XTTS config unpickling: `torch.serialization.add_safe_globals([xtts_config.XttsConfig])`
- This is normal and required for the model to load correctly
- No action needed if you see related warnings

### Getting Help

If you encounter issues:
1. Check that all dependencies are installed
2. Verify FFmpeg is installed and accessible
3. Ensure your voice sample meets the requirements
4. Review the error messages for specific guidance

## 📊 How It Works

1. **Voice Sample Processing**: Your voice sample is analyzed and optimized
   - Converted to mono audio
   - Normalized and compressed
   - Trimmed to optimal length
   - Creates temporary preprocessed file (`temp_preprocessed_voice.wav`) which is auto-deleted after processing

2. **Model Loading**: XTTS v2 model is loaded
   - **First run**: Downloads model from Hugging Face (~1.5GB, requires internet)
   - **Subsequent runs**: Loads from local cache (`~/.local/share/tts/`)
   - Uses GPU if available for faster processing
   - Model name: `tts_models/multilingual/multi-dataset/xtts_v2`
   - Includes PyTorch serialization workaround for XTTS config compatibility

3. **Voice Cloning**: The model generates speech
   - Analyzes your voice characteristics
   - Matches prosody and intonation
   - Generates natural-sounding speech
   - Uses optimized quality parameters for maximum similarity

4. **Output**: Saves the generated audio as WAV file
   - Default output: `output_max_quality.wav`
   - Format: WAV, 22050 Hz sample rate, mono

## 🎨 Quality Settings

The script uses maximum quality settings by default:
- **Ultra-low temperature** (0.05) for exact voice matching
- **High repetition penalty** (10.0) for natural speech
- **Optimized speed** (0.98) for clear articulation
- **Automatic preprocessing** for best input quality

## 📝 Example Output

After running the script, you'll see:
```
🎙️  Maximum Quality Voice Cloning Started...
📝 Text: Hi, I'm Nithin—a professional overthinker...
🔊 Voice sample: myvoice.wav
🔧 Preprocessing audio for maximum quality...
   ✓ Converted to mono
   ✓ Optimized sample rate
   ✓ Normalized volume
   ✓ Applied dynamic compression
   ✓ Trimmed edges
⏳ Loading AI model (this may take a moment)...
   🚀 GPU detected - using hardware acceleration
🎨 Generating speech with cloned voice...
✅ Success! Audio saved to: output_max_quality.wav
   📊 Duration: 12.34 seconds
```

## 🔒 Privacy & Ethics

**Important Considerations:**
- Only clone voices with explicit permission
- Do not use for impersonation or fraud
- Respect privacy and consent
- Use responsibly and ethically

**Data Privacy:**
- All processing happens locally on your machine
- No data is sent to external servers (except initial model download from Hugging Face)
- Voice samples and generated audio remain on your local system
- Models are cached locally and don't require internet after first download

## 📄 License

This project uses the Coqui TTS library, which is licensed under the MPL 2.0 license. Please refer to the [Coqui TTS license](https://github.com/coqui-ai/TTS/blob/main/LICENSE) for details.

## 🙏 Acknowledgments

- [Coqui TTS](https://github.com/coqui-ai/TTS) - The amazing TTS library
- XTTS v2 model developers
- PyTorch team

## 📚 Additional Resources

- [Coqui TTS Documentation](https://tts.readthedocs.io/)
- [XTTS Model Details](https://github.com/coqui-ai/TTS/wiki/XTTS-v2)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Hugging Face Model Hub](https://huggingface.co/models) - Source of XTTS v2 model

## 🔑 Important Technical Details

### Model Information
- **Model Name**: `tts_models/multilingual/multi-dataset/xtts_v2`
- **Model Size**: ~1.5GB
- **Storage Location**: 
  - Linux/macOS: `~/.local/share/tts/`
  - Windows: `%USERPROFILE%\.local\share\tts\`
- **Download Source**: Hugging Face (public repository, no authentication)
- **First Download**: Automatic on first run (requires internet)
- **Offline Use**: Works offline after initial download

### Temporary Files
- `temp_preprocessed_voice.wav` - Created during preprocessing, auto-deleted after use
- If script crashes, you may need to manually delete this file

### System Requirements Details
- **Internet**: Required only for first-time model download
- **Disk Space**: ~2GB for model + dependencies
- **RAM**: 4GB+ recommended (8GB+ for better performance)
- **VRAM**: 4GB+ if using GPU acceleration
- **CPU**: Multi-core recommended for faster processing

### No API Keys Required
- This project uses open-source models from Hugging Face
- No registration, API keys, or authentication needed
- All processing is local and private

---

**Enjoy creating amazing voice clones! 🎉**

