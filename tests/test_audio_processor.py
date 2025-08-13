#!/usr/bin/env python3
"""
Test the audio processor functions without requiring real audio hardware.
"""

import os
import sys
import numpy as np
import tempfile
from pathlib import Path

# Add parent directory to the path to import audiolog
sys.path.insert(0, str(Path(__file__).parent.parent))

from audiolog.audio.processor import AudioNormalizer, AudioEncoder


def create_synthetic_audio():
    """
    Create synthetic audio data for testing.
    
    Returns:
        NumPy array with a sine wave
    """
    # Create a 1-second sine wave at 440 Hz
    sample_rate = 16000
    duration = 1.0  # seconds
    
    # Generate array of time points
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Generate quiet sine wave (to test normalization)
    amplitude = 1000  # Quiet amplitude (16-bit audio has range of -32768 to 32767)
    audio = (amplitude * np.sin(2 * np.pi * 440 * t)).astype(np.int16)
    
    return audio


def test_audio_normalizer():
    """Test the audio normalization functionality."""
    print("Testing audio normalization...")
    
    # Create synthetic audio
    audio = create_synthetic_audio()
    
    # Get original amplitude stats
    original_max = np.max(np.abs(audio))
    original_rms = np.sqrt(np.mean(np.square(audio.astype(np.float32))))
    print(f"Original audio - Max amplitude: {original_max}, RMS: {original_rms:.2f}")
    
    # Create normalizer with target of -10 dB
    normalizer = AudioNormalizer(target_db=-10.0)
    
    # Normalize the audio
    normalized = normalizer.normalize(audio)
    
    # Get normalized amplitude stats
    normalized_max = np.max(np.abs(normalized))
    normalized_rms = np.sqrt(np.mean(np.square(normalized.astype(np.float32))))
    print(f"Normalized audio - Max amplitude: {normalized_max}, RMS: {normalized_rms:.2f}")
    
    # Check if normalization increased amplitude
    if normalized_rms > original_rms:
        print("Normalization successfully increased amplitude")
    else:
        print("ERROR: Normalization did not increase amplitude")
    
    return True


def test_audio_encoder():
    """Test the audio encoding functionality."""
    print("Testing audio encoding...")
    
    try:
        # Create synthetic audio
        audio = create_synthetic_audio()
        
        # Create encoder
        encoder = AudioEncoder(sample_rate=16000, channels=1)
        
        # Create a temporary directory for output
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "test.opus")
            
            # Encode to Opus
            encoded_path = encoder.pcm_to_opus(audio, output_path)
            
            # Check if file exists
            if os.path.exists(encoded_path):
                file_size = os.path.getsize(encoded_path)
                print(f"Successfully encoded audio to {encoded_path} (size: {file_size} bytes)")
            else:
                print(f"ERROR: Failed to encode audio, file not found: {encoded_path}")
            
            # Try normalizing and encoding
            processed_path = encoder.process_audio(
                audio,
                output_path=os.path.join(temp_dir, "normalized.opus"),
                normalize=True
            )
            
            if os.path.exists(processed_path):
                proc_file_size = os.path.getsize(processed_path)
                print(f"Successfully normalized and encoded audio to {processed_path} (size: {proc_file_size} bytes)")
            else:
                print(f"ERROR: Failed to process audio, file not found: {processed_path}")
        
        return True
    
    except Exception as e:
        print(f"Error in audio encoder test: {e}")
        return False


def main():
    """Main entry point."""
    print("=== AudioLog Audio Processor Tests ===\n")
    
    # Run tests
    test_audio_normalizer()
    print("\n")
    test_audio_encoder()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())