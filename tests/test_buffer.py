#!/usr/bin/env python3
"""
Simple test for the Audio Buffer module.
"""

import os
import sys
import time
import logging
import argparse
import numpy as np
from pathlib import Path

# Add parent directory to the path to import audiolog
sys.path.insert(0, str(Path(__file__).parent.parent))

from audiolog.core.vad import VoiceDetector, MicrophoneAudioStream, FileAudioStream
from audiolog.core.buffer import AudioBuffer, AudioProcessor


def setup_logging(verbose=False):
    """Set up basic logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )


def test_buffer_synthetic():
    """Test buffer with synthetic data."""
    print("Testing buffer with synthetic data...")
    
    # Create buffer with small settings for testing
    buffer = AudioBuffer(
        pre_buffer_seconds=1.0,  # 1 second
        post_buffer_seconds=1.0,  # 1 second
        sample_rate=16000,
        frame_duration_ms=30
    )
    
    processor = AudioProcessor(
        audio_buffer=buffer,
        sample_rate=16000,
        channels=1
    )
    
    # Create synthetic frames (30ms each at 16kHz = 480 samples = 960 bytes for 16-bit)
    frame_samples = 480
    frame_bytes = frame_samples * 2  # 16-bit = 2 bytes per sample
    
    # Generate silence
    silence_frame = np.zeros(frame_samples, dtype=np.int16).tobytes()
    
    # Generate tone (sine wave)
    def generate_tone_frame(freq=440, amplitude=10000):
        t = np.linspace(0, 0.03, frame_samples, endpoint=False)
        tone = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.int16)
        return tone.tobytes()
    
    tone_frame = generate_tone_frame()
    
    # Test pattern: 1s silence, 1s tone, 1s silence
    frames = []
    frames.extend([silence_frame] * 33)  # ~1s of silence (33 frames * 30ms = 990ms)
    frames.extend([tone_frame] * 33)     # ~1s of tone
    frames.extend([silence_frame] * 33)  # ~1s of silence
    
    # Process the frames
    segment = None
    speech_detected = False
    
    for i, frame in enumerate(frames):
        is_speech = (tone_frame == frame)  # Treat tone frames as speech
        
        # Log transitions
        if is_speech and not speech_detected:
            print(f"Speech starts at frame {i}")
            speech_detected = True
        elif not is_speech and speech_detected:
            print(f"Speech ends at frame {i}")
            speech_detected = False
        
        # Add to buffer
        result = buffer.add_frame(frame, is_speech)
        
        if result is not None:
            segment = result
            print(f"Complete segment received with {len(segment)} frames")
    
    # Verify results
    if segment is None:
        print("ERROR: No segment was returned")
        return
    
    # Expected segment size = 1s pre + 1s tone + 1s post = ~3s = ~100 frames
    expected_frames = 100
    margin = 5  # Allow for small timing differences
    
    if abs(len(segment) - expected_frames) > margin:
        print(f"WARNING: Unexpected segment size: {len(segment)} frames (expected ~{expected_frames})")
    else:
        print(f"Segment size matches expected: {len(segment)} frames")
    
    # Convert segment to PCM and then to numpy for analysis
    pcm_data = processor.frames_to_pcm(segment)
    audio_array = processor.pcm_to_numpy(pcm_data)
    
    # Analyze the segment to find where the tone starts/ends
    is_tone = np.abs(audio_array) > 1000
    tone_starts = np.where(np.diff(is_tone.astype(int)) == 1)[0]
    tone_ends = np.where(np.diff(is_tone.astype(int)) == -1)[0]
    
    if len(tone_starts) > 0 and len(tone_ends) > 0:
        tone_start_frame = tone_starts[0] // frame_samples
        tone_end_frame = tone_ends[-1] // frame_samples
        print(f"Tone detected from frame {tone_start_frame} to {tone_end_frame}")
        
        # Check pre-buffer (should be ~1s before tone starts)
        pre_buffer_frames = tone_start_frame
        print(f"Pre-buffer: {pre_buffer_frames} frames (~{pre_buffer_frames * 0.03:.2f}s)")
        
        # Check post-buffer (should be ~1s after tone ends)
        post_buffer_frames = len(segment) - tone_end_frame - 1
        print(f"Post-buffer: {post_buffer_frames} frames (~{post_buffer_frames * 0.03:.2f}s)")
    else:
        print("WARNING: Could not detect tone in segment")


def test_buffer_with_mic(duration=10, pre_buffer_sec=2, post_buffer_sec=2):
    """
    Test buffer with microphone input.
    
    Args:
        duration: Test duration in seconds
        pre_buffer_sec: Pre-buffer duration in seconds
        post_buffer_sec: Post-buffer duration in seconds
    """
    print(f"Testing buffer with microphone for {duration}s (pre={pre_buffer_sec}s, post={post_buffer_sec}s)...")
    
    # Create VAD
    vad = VoiceDetector(
        aggressiveness=3,
        sample_rate=16000,
        frame_duration_ms=30
    )
    
    # Create buffer
    buffer = AudioBuffer(
        pre_buffer_seconds=pre_buffer_sec,
        post_buffer_seconds=post_buffer_sec,
        sample_rate=16000,
        frame_duration_ms=30
    )
    
    processor = AudioProcessor(
        audio_buffer=buffer,
        sample_rate=16000,
        channels=1
    )
    
    # Create microphone stream
    with MicrophoneAudioStream(
        sample_rate=16000,
        channels=1,
        chunk_size=480  # 30ms at 16kHz
    ) as mic:
        # Get the stream
        stream = mic.get_stream()
        
        # Process audio frames
        start_time = time.time()
        segments = []
        
        print("Speak to test the buffer...")
        
        while time.time() - start_time < duration:
            try:
                # Get frame
                frame = next(stream)
                
                # Detect speech
                is_speech = vad.is_speech(frame)
                
                # Add to buffer
                segment = buffer.add_frame(frame, is_speech)
                
                if segment is not None:
                    segments.append(segment)
                    segment_duration = processor.get_duration_seconds(segment)
                    print(f"Segment complete! Duration: {segment_duration:.2f}s, Frames: {len(segment)}")
            
            except StopIteration:
                break
        
        # Print results
        print(f"Test complete. Captured {len(segments)} segments")
        
        for i, segment in enumerate(segments):
            segment_duration = processor.get_duration_seconds(segment)
            pcm_data = processor.frames_to_pcm(segment)
            audio_array = processor.pcm_to_numpy(pcm_data)
            
            # Simple analysis of the segment
            audio_max = np.max(np.abs(audio_array))
            audio_rms = np.sqrt(np.mean(np.square(audio_array.astype(np.float32))))
            
            print(f"  Segment {i+1}: {segment_duration:.2f}s, {len(segment)} frames, "
                  f"Max amplitude: {audio_max}, RMS: {audio_rms:.2f}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Test Audio Buffer'
    )
    
    # Test type
    test_type = parser.add_mutually_exclusive_group(required=True)
    test_type.add_argument(
        '--synthetic', 
        action='store_true',
        help='Test with synthetic data'
    )
    test_type.add_argument(
        '--mic',
        action='store_true',
        help='Test with microphone input'
    )
    
    # Options
    parser.add_argument(
        '-d', '--duration',
        type=int,
        default=10,
        help='Test duration in seconds for microphone test (default: 10)'
    )
    parser.add_argument(
        '--pre',
        type=float,
        default=2.0,
        help='Pre-buffer duration in seconds (default: 2.0)'
    )
    parser.add_argument(
        '--post',
        type=float,
        default=2.0,
        help='Post-buffer duration in seconds (default: 2.0)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)
    
    try:
        if args.synthetic:
            test_buffer_synthetic()
        elif args.mic:
            test_buffer_with_mic(args.duration, args.pre, args.post)
            
    except KeyboardInterrupt:
        print("\nTest interrupted")
    except Exception as e:
        print(f"Error: {e}")
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main())