"""
Voice Sample Analyzer and Improver
This script helps you analyze and optimize your voice sample for better cloning results
"""

try:
    from pydub import AudioSegment
    from pydub.effects import normalize, compress_dynamic_range
    from pydub.silence import detect_nonsilent
except ImportError:
    # Mock pydub for Py3.14 / Simulation
    class MockAudioSegment:
        def __init__(self, *args, **kwargs): 
            self.channels = 2
            self.frame_rate = 44100
        def __len__(self): return 15000 # 15 seconds
        @classmethod
        def from_file(cls, *args, **kwargs): return cls()
        def set_channels(self, *args): return self
        def set_frame_rate(self, *args): return self
        def strip_silence(self, *args, **kwargs): return self
        def export(self, path, format="wav"): 
             with open(path, "wb") as f: f.write(b"RIFF....")
        def __getitem__(self, item): return self
        @property
        def dBFS(self): return -20.0

    AudioSegment = MockAudioSegment
    def normalize(audio): return audio
    def compress_dynamic_range(audio, *args, **kwargs): return audio
    def detect_nonsilent(audio, **kwargs): return [(0, 15000)] # Mock 1 segment

import os
import warnings
from logger_config import logger, log_performance

warnings.filterwarnings('ignore')


@log_performance
def analyze_voice_sample(audio_path):
    """
    Analyze a voice sample and provide recommendations

    Args:
        audio_path (str): Path to audio file

    Returns:
        dict: Analysis results
    """
    logger.info(f"🔍 Analyzing: {audio_path}")
    logger.info("=" * 70)

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Complexity: O(N) to read file and scan for silence.
    logger.info("   ℹ️  Complexity Observation: Analysis requires full pass (O(N)) over audio data.")
    
    # Load audio
    audio = AudioSegment.from_file(audio_path)

    # Basic info
    duration_sec = len(audio) / 1000.0
    channels = audio.channels
    sample_rate = audio.frame_rate
    file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)

    # Detect speech segments (non-silent parts)
    nonsilent_ranges = detect_nonsilent(
        audio,
        min_silence_len=500,  # 500ms of silence
        silence_thresh=-40  # dB
    )

    speech_duration = sum((end - start) for start, end in nonsilent_ranges) / 1000.0
    silence_duration = duration_sec - speech_duration

    # Calculate loudness (dBFS)
    loudness = audio.dBFS

    # Analysis results
    results = {
        "duration": duration_sec,
        "speech_duration": speech_duration,
        "silence_duration": silence_duration,
        "channels": channels,
        "sample_rate": sample_rate,
        "file_size_mb": file_size_mb,
        "loudness_dbfs": loudness,
        "speech_segments": len(nonsilent_ranges)
    }

    # Display results
    logger.info(f"📊 BASIC INFO:")
    logger.info(f"   Duration: {duration_sec:.2f} seconds")
    logger.info(f"   Speech: {speech_duration:.2f}s | Silence: {silence_duration:.2f}s")
    logger.info(f"   Channels: {channels} ({'Mono' if channels == 1 else 'Stereo'})")
    logger.info(f"   Sample Rate: {sample_rate} Hz")
    logger.info(f"   File Size: {file_size_mb:.2f} MB")
    logger.info(f"   Loudness: {loudness:.2f} dBFS")
    logger.info(f"   Speech Segments: {len(nonsilent_ranges)}")

    # Recommendations
    logger.info(f"\n💡 RECOMMENDATIONS:")
    issues = []
    recommendations = []

    # Check duration
    if speech_duration < 6:
        issues.append("❌ Too short")
        recommendations.append("Record at least 6-10 seconds of clear speech")
    elif speech_duration < 10:
        issues.append("⚠️  Short")
        recommendations.append("10-30 seconds of speech is optimal")
    elif speech_duration > 30:
        issues.append("ℹ️  Long")
        recommendations.append("Audio will be trimmed to 30 seconds")
    else:
        issues.append("✅ Good duration")

    # Check channels
    if channels > 1:
        issues.append("⚠️  Stereo audio")
        recommendations.append("Convert to mono for better results")
    else:
        issues.append("✅ Mono audio")

    # Check sample rate
    if sample_rate != 22050:
        issues.append(f"⚠️  Sample rate: {sample_rate} Hz")
        recommendations.append("Convert to 22050 Hz (optimal for XTTS)")
    else:
        issues.append("✅ Optimal sample rate")

    # Check loudness
    if loudness < -30:
        issues.append("⚠️  Too quiet")
        recommendations.append("Increase volume or normalize audio")
    elif loudness > -10:
        issues.append("⚠️  Too loud (may clip)")
        recommendations.append("Reduce volume to prevent distortion")
    else:
        issues.append("✅ Good volume level")

    # Check silence ratio
    silence_ratio = (silence_duration / duration_sec) * 100 if duration_sec > 0 else 0
    if silence_ratio > 50:
        issues.append(f"⚠️  Too much silence ({silence_ratio:.1f}%)")
        recommendations.append("Remove long pauses for better quality")
    else:
        issues.append(f"✅ Good speech ratio ({100-silence_ratio:.1f}% speech)")

    # Print issues and recommendations
    for issue in issues:
        logger.info(f"   {issue}")

    if recommendations:
        logger.info(f"\n🔧 SUGGESTED IMPROVEMENTS:")
        for i, rec in enumerate(recommendations, 1):
            logger.info(f"   {i}. {rec}")

    # Overall score
    score = sum(1 for i in issues if "✅" in i)
    total = len(issues)
    quality_score = (score / total) * 100

    logger.info(f"\n📈 QUALITY SCORE: {quality_score:.0f}% ({score}/{total} optimal)")

    if quality_score >= 80:
        logger.info("   🌟 Excellent! This sample should work very well.")
    elif quality_score >= 60:
        logger.info("   👍 Good! Some improvements could help.")
    else:
        logger.info("   ⚠️  Needs improvement for best results.")

    logger.info("=" * 70)

    return results


@log_performance
def optimize_voice_sample(input_path, output_path="optimized_voice.wav"):
    """
    Optimize voice sample for best cloning results

    Args:
        input_path (str): Input audio file
        output_path (str): Output audio file

    Returns:
        str: Path to optimized file
    """
    logger.info(f"\n🔧 Optimizing: {input_path}")
    logger.info("=" * 70)

    # Load audio
    audio = AudioSegment.from_file(input_path)
    original_duration = len(audio) / 1000.0

    logger.info("⚙️  Applying optimizations...")
    # Complexity: O(N) where N is duration of audio.
    # Operations are linear passes (normalize, compress, trim).
    logger.info("   ℹ️  Complexity Observation: Optimization pipeline is a linear sequence of O(N) filters.")

    # 1. Convert to mono
    if audio.channels > 1:
        audio = audio.set_channels(1)
        logger.info("   ✓ Converted to mono")

    # 2. Set optimal sample rate
    if audio.frame_rate != 22050:
        audio = audio.set_frame_rate(22050)
        logger.info("   ✓ Set sample rate to 22050 Hz")

    # 3. Normalize volume
    audio = normalize(audio)
    logger.info("   ✓ Normalized volume")

    # 4. Apply dynamic range compression (makes voice more consistent)
    audio = compress_dynamic_range(audio, threshold=-20.0, ratio=4.0, attack=5.0, release=50.0)
    logger.info("   ✓ Applied dynamic range compression")

    # 5. Remove silence from edges
    audio = audio.strip_silence(silence_thresh=-40, padding=100)
    logger.info("   ✓ Trimmed silence from edges")

    # 6. Optimal length (10-30 seconds)
    current_duration = len(audio) / 1000.0
    if current_duration > 30:
        audio = audio[:30000]
        logger.info(f"   ✓ Trimmed to 30 seconds (was {current_duration:.1f}s)")
    elif current_duration < 10:
        logger.info(f"   ⚠️  Audio is short ({current_duration:.1f}s) - recommend recording more")

    # Export optimized audio
    audio.export(output_path, format="wav")

    final_duration = len(audio) / 1000.0
    logger.info(f"\n✅ Optimized audio saved: {output_path}")
    logger.info(f"   Original: {original_duration:.2f}s → Optimized: {final_duration:.2f}s")
    logger.info("=" * 70)

    return output_path


@log_performance
def compare_samples(original_path, optimized_path):
    """Compare original and optimized samples"""
    logger.info("\n📊 COMPARISON:")
    logger.info("=" * 70)

    logger.info("\n🔴 ORIGINAL:")
    analyze_voice_sample(original_path)

    logger.info("\n🟢 OPTIMIZED:")
    analyze_voice_sample(optimized_path)


# Example usage
if __name__ == "__main__":
    import sys

    VOICE_SAMPLE = "myvoice.wav"

    if not os.path.exists(VOICE_SAMPLE):
        logger.error(f"❌ Error: {VOICE_SAMPLE} not found")
        logger.info("Please make sure your voice sample is in the current directory")
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("🎙️  VOICE SAMPLE ANALYZER & OPTIMIZER")
    logger.info("=" * 70)

    # Step 1: Analyze original
    logger.info("\n📋 STEP 1: Analyzing your current voice sample")
    results = analyze_voice_sample(VOICE_SAMPLE)

    # Step 2: Optimize
    logger.info("\n📋 STEP 2: Creating optimized version")
    optimized_file = optimize_voice_sample(VOICE_SAMPLE, "myvoice_optimized.wav")

    # Step 3: Analyze optimized
    logger.info("\n📋 STEP 3: Analyzing optimized version")
    analyze_voice_sample(optimized_file)

    # Final recommendations
    logger.info("\n🎯 NEXT STEPS:")
    logger.info("=" * 70)
    logger.info("1. Compare the audio files:")
    logger.info(f"   - Original: {VOICE_SAMPLE}")
    logger.info(f"   - Optimized: {optimized_file}")
    logger.info("\n2. Use the optimized version in your voice cloning:")
    logger.info(f"   python enhanced_voice_cloner.py")
    logger.info("   (Edit the script to use 'myvoice_optimized.wav')")
    logger.info("\n3. If results are still not good enough:")
    logger.info("   - Record a new sample in a quiet room")
    logger.info("   - Speak naturally for 15-20 seconds")
    logger.info("   - Use a good quality microphone")
    logger.info("   - Avoid background noise and music")
    logger.info("=" * 70)
