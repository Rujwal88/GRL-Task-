
# Setup environment for Qwen3 TTS Voice Cloning

Write-Host "Checking ffmpeg..."
if (Test-Path ".\ffmpeg_bin") {
    $env:PATH += ";$PWD\ffmpeg_bin"
    Write-Host "Added ffmpeg_bin to PATH"
}

Write-Host "Upgrading pip..."
python -m pip install --upgrade pip

Write-Host "Installing core dependencies..."
python -m pip install torch torchaudio transformers accelerate pydub

Write-Host "Attempting to install qwentts3 package..."
try {
    python -m pip install qwentts3
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Successfully installed qwentts3 package."
    } else {
        throw "Package install failed"
    }
} catch {
    Write-Host "qwentts3 package not found on PyPI. Falling back to GitHub clone..."
    if (-not (Test-Path "Qwen3-TTS")) {
        git clone https://github.com/QwenLM/Qwen3-TTS.git
    }
    if (Test-Path "Qwen3-TTS") {
        cd Qwen3-TTS
        python -m pip install -r requirements.txt
        python -m pip install .
        cd ..
    } else {
        Write-Host "Failed to clone Qwen3-TTS repository. Trying alternative..."
        # Maybe Qwen/Qwen3-TTS?
        git clone https://github.com/Qwen/Qwen3-TTS.git
        if (Test-Path "Qwen3-TTS") {
             cd Qwen3-TTS
             python -m pip install -r requirements.txt
             python -m pip install .
             cd ..
        } else {
             Write-Host "CRITICAL: Could not find Qwen3-TTS repository!"
        }
    }
}

Write-Host "Verifying torch..."
python -c "import torch; print(f'Torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
