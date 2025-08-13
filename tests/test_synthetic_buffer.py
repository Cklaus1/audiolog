#!/usr/bin/env python3
"""
Test the audio buffer with synthetic data, without requiring PyAudio.
"""

import os
import sys
import time
import logging
import numpy as np
from pathlib import Path

# Add parent directory to the path to import audiolog
sys.path.insert(0, str(Path(__file__).parent.parent))

from audiolog.core.buffer import AudioBuffer, AudioProcessor


def setup_logging(verbose=False):
    """Set up basic logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

def create_synthetic_audio_frames(
    duration_seconds=3.0,  # Total duration
    sample_rate=16000,     # Samples per second
    frame_duration_ms=30,  # Frame duration in milliseconds
    has_speech=True,       # Whether to include speech
    speech_start_sec=1.0,  # When speech starts
    speech_end_sec=2.0     # When speech ends
):
    """
    Create synthetic audio frames for testing.
    
    Args:
        duration_seconds: Total duration in seconds
        sample_rate: Sample rate in Hz
        frame_duration_ms: Frame duration in milliseconds
        has_speech: Whether to include speech
        speech_start_sec: When speech starts (seconds)
        speech_end_sec: When speech ends (seconds)
        
    Returns:
        Tuple of (frames, speech_mask)
    """
    # Calculate parameters
    frame_size = int(sample_rate * frame_duration_ms / 1000)  # samples per frame
    frame_count = int(duration_seconds * 1000 / frame_duration_ms)
    
    # Create frames
    frames = []
    speech_mask = []
    
    # Generate frames
    for i in range(frame_count):
        frame_time = i * frame_duration_ms / 1000  # Time in seconds
        
        # Determine if this frame has speech
        is_speech = has_speech and (speech_start_sec <= frame_time < speech_end_sec)
        speech_mask.append(is_speech)
        
        if is_speech:
            # Generate tone (sine wave) for speech
            t = np.linspace(0, frame_duration_ms/1000, frame_size, endpoint=False)
            tone = (10000 * np.sin(2 * np.pi * 440 * t)).astype(np.int16)
            frame_data = tone.tobytes()
        else:
            # Generate silence
            frame_data = np.zeros(frame_size, dtype=np.int16).tobytes()
        
        frames.append(frame_data)
    
    return frames, speech_mask


def test_buffer_with_synthetic_audio():
    """Test the buffer with synthetic audio data."""
    print("Testing buffer with synthetic audio data...")
    
    # Create buffer with smaller settings for testing
    buffer = AudioBuffer(
        pre_buffer_seconds=0.5,  # 0.5 seconds
        post_buffer_seconds=0.5,  # 0.5 seconds
        sample_rate=16000,
        frame_duration_ms=30
    )
    
    processor = AudioProcessor(
        audio_buffer=buffer,
        sample_rate=16000,
        channels=1
    )
    
    # Create synthetic audio
    frames, speech_mask = create_synthetic_audio_frames(
        duration_seconds=3.0,
        sample_rate=16000,
        frame_duration_ms=30,
        has_speech=True,
        speech_start_sec=1.0,
        speech_end_sec=2.0
    )
    
    # Process frames
    segment = None
    for i, (frame, is_speech) in enumerate(zip(frames, speech_mask)):
        if is_speech and not buffer.is_speech_active:
            print(f"Frame {i}: Speech started")
        elif not is_speech and buffer.is_speech_active:
            print(f"Frame {i}: Speech ended")
            
        # Add to buffer
        result = buffer.add_frame(frame, is_speech)
        
        if result is not None:
            segment = result
            print(f"Frame {i}: Complete segment received with {len(segment)} frames")
    
    # Verify results
    if segment is None:
        print("ERROR: No segment was returned")
        return False
    
    # Convert segment to PCM and then to numpy for analysis
    pcm_data = processor.frames_to_pcm(segment)
    audio_array = processor.pcm_to_numpy(pcm_data)
    
    # Print segment details
    segment_duration = processor.get_duration_seconds(segment)
    print(f"Segment duration: {segment_duration:.3f} seconds")
    print(f"Segment frames: {len(segment)}")
    
    # Analyze the segment
    rms = np.sqrt(np.mean(np.square(audio_array.astype(np.float32))))
    max_amplitude = np.max(np.abs(audio_array))
    print(f"RMS amplitude: {rms:.2f}")
    print(f"Max amplitude: {max_amplitude}")
    
    # Check if segment contains speech
    is_tone = np.abs(audio_array) > 1000
    has_tone = np.any(is_tone)
    print(f"Segment contains speech: {has_tone}")
    
    return True


def main():
    """Main entry point."""
    print("=== AudioLog Synthetic Buffer Test ===\n")
    
    setup_logging(verbose=False)
    test_buffer_with_synthetic_audio()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())