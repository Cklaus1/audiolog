#!/usr/bin/env python3
"""
AudioLog Full Demo Script

This script demonstrates the core functionality of AudioLog by simulating
a complete audio processing cycle without requiring microphone access.
"""

import os
import sys
import time
import logging
import numpy as np
import tempfile
from datetime import datetime
from pathlib import Path

# Add main directory to import path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import AudioLog components
from audiolog.utils.config import config
from audiolog.core.buffer import AudioBuffer, AudioProcessor
from audiolog.audio.processor import AudioNormalizer, AudioEncoder
from audiolog.audio.storage import AudioStorage

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("audiolog-demo")

def generate_synthetic_audio(
    duration=5.0,
    sample_rate=16000,
    has_speech=True,
    speech_start_sec=1.5,
    speech_end_sec=3.5,
    amplitude=10000
):
    """
    Generate synthetic audio with speech in the middle.
    
    Args:
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        has_speech: Whether to include speech
        speech_start_sec: When speech starts
        speech_end_sec: When speech ends
        amplitude: Amplitude of speech
        
    Returns:
        NumPy array of audio samples
    """
    # Create time array
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Create an array of zeros (silence)
    audio = np.zeros(len(t), dtype=np.int16)
    
    if has_speech:
        # Add speech in the specified segment
        speech_start_idx = int(speech_start_sec * sample_rate)
        speech_end_idx = int(speech_end_sec * sample_rate)
        
        # Make sure indices are valid
        if speech_start_idx >= len(audio):
            speech_start_idx = 0
        if speech_end_idx >= len(audio):
            speech_end_idx = len(audio) - 1
            
        # Generate speech signal (sine wave at 440Hz)
        speech_t = t[speech_start_idx:speech_end_idx]
        speech = (amplitude * np.sin(2 * np.pi * 440 * speech_t)).astype(np.int16)
        
        # Insert speech into silence
        audio[speech_start_idx:speech_end_idx] = speech
    
    return audio

def demo_audio_processing():
    """
    Demonstrate audio processing components of AudioLog.
    
    Creates synthetic audio, normalizes it, encodes to Opus,
    and saves it with metadata.
    """
    logger.info("=== AudioLog Audio Processing Demo ===")
    
    # Generate synthetic audio
    sample_rate = 16000
    audio = generate_synthetic_audio(
        duration=5.0,
        sample_rate=sample_rate,
        has_speech=True
    )
    
    logger.info(f"Generated 5 seconds of synthetic audio at {sample_rate}Hz")
    
    # Create components
    normalizer = AudioNormalizer(target_db=-16.0)
    encoder = AudioEncoder(sample_rate=sample_rate, channels=1)
    storage = AudioStorage("audiolog_demo_output")
    
    # Normalize audio
    logger.info("Normalizing audio...")
    normalized = normalizer.normalize(audio)
    
    # Create output directory if needed
    output_dir = "audiolog_demo_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Encode to Opus
    logger.info("Encoding to Opus format...")
    opus_path = os.path.join(output_dir, "synthetic_speech.opus")
    encoder.pcm_to_opus(normalized, opus_path)
    
    # Create metadata
    now = time.time()
    metadata = {
        "device_id": "demo_device",
        "start_time": now - 5.0,  # 5 seconds ago
        "end_time": now,
        "duration": 5.0,
        "sample_rate": sample_rate,
        "channels": 1,
        "normalized": True,
        "demo": True
    }
    
    # Save to storage
    logger.info("Saving to storage with metadata...")
    audio_path, metadata_path = storage.save_audio_file(
        opus_path,
        metadata,
        device_id="demo_device",
        timestamp=datetime.fromtimestamp(now - 5.0)
    )
    
    logger.info(f"Saved audio file: {audio_path}")
    logger.info(f"Saved metadata: {metadata_path}")
    
    return audio_path, metadata_path

def demo_buffer_system():
    """
    Demonstrate the audio buffer system.
    
    Simulates audio frames coming in, with speech detection,
    and shows how the buffer captures speech with pre/post context.
    """
    logger.info("\n=== AudioLog Buffer System Demo ===")
    
    # Create a buffer with small durations for demo
    pre_buffer_sec = 1.0
    post_buffer_sec = 1.0
    sample_rate = 16000
    frame_ms = 30
    
    buffer = AudioBuffer(
        pre_buffer_seconds=pre_buffer_sec,
        post_buffer_seconds=post_buffer_sec,
        sample_rate=sample_rate,
        frame_duration_ms=frame_ms
    )
    
    processor = AudioProcessor(
        audio_buffer=buffer,
        sample_rate=sample_rate,
        channels=1
    )
    
    logger.info(f"Created buffer with {pre_buffer_sec}s pre-buffer and {post_buffer_sec}s post-buffer")
    
    # Calculate frames per second and frame size
    frames_per_sec = 1000 / frame_ms
    frame_size = int(sample_rate * frame_ms / 1000)
    
    # Create synthetic frames to simulate a stream
    duration_sec = 5.0
    frame_count = int(duration_sec * frames_per_sec)
    
    # Determine which frames have speech (middle third)
    speech_start_frame = frame_count // 3
    speech_end_frame = (frame_count * 2) // 3
    
    logger.info(f"Simulating {frame_count} frames ({duration_sec}s) with speech from "
              f"frames {speech_start_frame}-{speech_end_frame}")
    
    # Generate and process frames
    segment = None
    
    for i in range(frame_count):
        # Determine if this frame has speech
        is_speech = (speech_start_frame <= i < speech_end_frame)
        
        # Generate frame data (silence or speech)
        if is_speech:
            # Create sine wave
            t = np.linspace(0, frame_ms/1000, frame_size, endpoint=False)
            audio = (10000 * np.sin(2 * np.pi * 440 * t)).astype(np.int16)
            frame_data = audio.tobytes()
        else:
            # Create silence
            frame_data = np.zeros(frame_size, dtype=np.int16).tobytes()
        
        # Log transitions
        if i == speech_start_frame:
            logger.info(f"Speech started at frame {i} (time: {i/frames_per_sec:.2f}s)")
        elif i == speech_end_frame:
            logger.info(f"Speech ended at frame {i} (time: {i/frames_per_sec:.2f}s)")
        
        # Add frame to buffer and check for completed segments
        result = buffer.add_frame(frame_data, is_speech)
        
        if result is not None:
            segment = result
            segment_frames = len(segment)
            segment_duration = processor.get_duration_seconds(segment)
            logger.info(f"Complete segment received: {segment_frames} frames, "
                      f"{segment_duration:.2f}s duration")
            
            # Analyze the segment
            pcm_data = processor.frames_to_pcm(segment)
            audio_array = processor.pcm_to_numpy(pcm_data)
            
            # Count zero and non-zero frames
            non_zero_samples = np.count_nonzero(audio_array)
            zero_samples = len(audio_array) - non_zero_samples
            logger.info(f"Segment analysis: {non_zero_samples} non-zero samples, "
                      f"{zero_samples} zero samples")
    
    if segment is None:
        logger.warning("No complete segment was returned - try increasing the duration")
    
    return segment

if __name__ == "__main__":
    # Run audio processing demo
    audio_path, metadata_path = demo_audio_processing()
    
    # Run buffer system demo
    segment = demo_buffer_system()
    
    logger.info("\nDemo complete!")