"""
Voice Sample Analyzer and Improver
This script helps you analyze and optimize your voice sample for better cloning results
"""

import os
from pydub import AudioSegment
from pydub.effects import normalize, compress_dynamic_range
from pydub.silence import detect_nonsilent
import warnings

warnings.filterwarnings('ignore')


def analyze_voice_sample(audio_path):
    """
    Analyze a voice sample and provide recommendations

    Args:
        audio_path (str): Path to audio file

    Returns:
        dict: Analysis results
    """
    print(f"🔍 Analyzing: {audio_path}")
    print("=" * 70)

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

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
    print(f"📊 BASIC INFO:")
    print(f"   Duration: {duration_sec:.2f} seconds")
    print(f"   Speech: {speech_duration:.2f}s | Silence: {silence_duration:.2f}s")
    print(f"   Channels: {channels} ({'Mono' if channels == 1 else 'Stereo'})")
    print(f"   Sample Rate: {sample_rate} Hz")
    print(f"   File Size: {file_size_mb:.2f} MB")
    print(f"   Loudness: {loudness:.2f} dBFS")
    print(f"   Speech Segments: {len(nonsilent_ranges)}")

    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
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
        print(f"   {issue}")

    if recommendations:
        print(f"\n🔧 SUGGESTED IMPROVEMENTS:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")

    # Overall score
    score = sum(1 for i in issues if "✅" in i)
    total = len(issues)
    quality_score = (score / total) * 100

    print(f"\n📈 QUALITY SCORE: {quality_score:.0f}% ({score}/{total} optimal)")

    if quality_score >= 80:
        print("   🌟 Excellent! This sample should work very well.")
    elif quality_score >= 60:
        print("   👍 Good! Some improvements could help.")
    else:
        print("   ⚠️  Needs improvement for best results.")

    print("=" * 70)

    return results


def optimize_voice_sample(input_path, output_path="optimized_voice.wav"):
    """
    Optimize voice sample for best cloning results

    Args:
        input_path (str): Input audio file
        output_path (str): Output audio file

    Returns:
        str: Path to optimized file
    """
    print(f"\n🔧 Optimizing: {input_path}")
    print("=" * 70)

    # Load audio
    audio = AudioSegment.from_file(input_path)
    original_duration = len(audio) / 1000.0

    print("⚙️  Applying optimizations...")

    # 1. Convert to mono
    if audio.channels > 1:
        audio = audio.set_channels(1)
        print("   ✓ Converted to mono")

    # 2. Set optimal sample rate
    if audio.frame_rate != 22050:
        audio = audio.set_frame_rate(22050)
        print("   ✓ Set sample rate to 22050 Hz")

    # 3. Normalize volume
    audio = normalize(audio)
    print("   ✓ Normalized volume")

    # 4. Apply dynamic range compression (makes voice more consistent)
    audio = compress_dynamic_range(audio, threshold=-20.0, ratio=4.0, attack=5.0, release=50.0)
    print("   ✓ Applied dynamic range compression")

    # 5. Remove silence from edges
    audio = audio.strip_silence(silence_thresh=-40, padding=100)
    print("   ✓ Trimmed silence from edges")

    # 6. Optimal length (10-30 seconds)
    current_duration = len(audio) / 1000.0
    if current_duration > 30:
        audio = audio[:30000]
        print(f"   ✓ Trimmed to 30 seconds (was {current_duration:.1f}s)")
    elif current_duration < 10:
        print(f"   ⚠️  Audio is short ({current_duration:.1f}s) - recommend recording more")

    # Export optimized audio
    audio.export(output_path, format="wav")

    final_duration = len(audio) / 1000.0
    print(f"\n✅ Optimized audio saved: {output_path}")
    print(f"   Original: {original_duration:.2f}s → Optimized: {final_duration:.2f}s")
    print("=" * 70)

    return output_path


def compare_samples(original_path, optimized_path):
    """Compare original and optimized samples"""
    print("\n📊 COMPARISON:")
    print("=" * 70)

    print("\n🔴 ORIGINAL:")
    analyze_voice_sample(original_path)

    print("\n🟢 OPTIMIZED:")
    analyze_voice_sample(optimized_path)


# Example usage
if __name__ == "__main__":
    import sys

    VOICE_SAMPLE = "myvoice.wav"

    if not os.path.exists(VOICE_SAMPLE):
        print(f"❌ Error: {VOICE_SAMPLE} not found")
        print("Please make sure your voice sample is in the current directory")
        sys.exit(1)

    print("=" * 70)
    print("🎙️  VOICE SAMPLE ANALYZER & OPTIMIZER")
    print("=" * 70)

    # Step 1: Analyze original
    print("\n📋 STEP 1: Analyzing your current voice sample")
    results = analyze_voice_sample(VOICE_SAMPLE)

    # Step 2: Optimize
    print("\n📋 STEP 2: Creating optimized version")
    optimized_file = optimize_voice_sample(VOICE_SAMPLE, "myvoice_optimized.wav")

    # Step 3: Analyze optimized
    print("\n📋 STEP 3: Analyzing optimized version")
    analyze_voice_sample(optimized_file)

    # Final recommendations
    print("\n🎯 NEXT STEPS:")
    print("=" * 70)
    print("1. Compare the audio files:")
    print(f"   - Original: {VOICE_SAMPLE}")
    print(f"   - Optimized: {optimized_file}")
    print("\n2. Use the optimized version in your voice cloning:")
    print(f"   python enhanced_voice_cloner.py")
    print("   (Edit the script to use 'myvoice_optimized.wav')")
    print("\n3. If results are still not good enough:")
    print("   - Record a new sample in a quiet room")
    print("   - Speak naturally for 15-20 seconds")
    print("   - Use a good quality microphone")
    print("   - Avoid background noise and music")
    print("=" * 70)
