#!/usr/bin/env python3
"""
AudioLog Recording Simulation

This script simulates continuous recording with the AudioLog system
by generating synthetic audio with occasional speech events.
"""

import os
import sys
import time
import random
import logging
import threading
import tempfile
import numpy as np
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
logger = logging.getLogger("audiolog-sim")

# Global state
RUNNING = True

def generate_frame(
    sample_rate=16000,
    frame_duration_ms=30,
    is_speech=False,
    frequency=440,
    amplitude=10000,
    noise_level=500
):
    """
    Generate a single audio frame.
    
    Args:
        sample_rate: Sample rate in Hz
        frame_duration_ms: Frame duration in milliseconds
        is_speech: Whether the frame contains speech
        frequency: Frequency of speech tone in Hz
        amplitude: Amplitude of speech
        noise_level: Background noise level
        
    Returns:
        Bytes containing the audio frame data
    """
    # Calculate frame size
    frame_size = int(sample_rate * frame_duration_ms / 1000)
    
    # Generate time array
    t = np.linspace(0, frame_duration_ms/1000, frame_size, endpoint=False)
    
    # Generate background noise (lower amplitude)
    noise = np.random.normal(0, noise_level, frame_size).astype(np.int16)
    
    if is_speech:
        # Generate speech signal (sine wave)
        speech = (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.int16)
        
        # Add some amplitude variation to make it more realistic
        mod = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)  # 3 Hz modulation
        speech = (speech * mod).astype(np.int16)
        
        # Add speech to noise
        audio = speech + noise
    else:
        # Just noise
        audio = noise
    
    return audio.tobytes()


def simulate_audio_stream(
    sample_rate=16000,
    frame_duration_ms=30,
    speech_pattern="random",  # "random", "periodic", or "triggered"
    speech_frequency=0.2,     # For random mode: probability of speech
    speech_period=5.0,        # For periodic mode: seconds between speech
    speech_duration=1.5       # Duration of speech events
):
    """
    Generate a simulated audio stream with speech events.
    
    Args:
        sample_rate: Sample rate in Hz
        frame_duration_ms: Frame duration in milliseconds
        speech_pattern: Pattern of speech events
        speech_frequency: Probability of speech (random mode)
        speech_period: Seconds between speech events (periodic mode)
        speech_duration: Duration of speech events in seconds
        
    Yields:
        Tuples of (frame_data, is_speech)
    """
    # Calculate frames per second
    frames_per_sec = 1000 / frame_duration_ms
    
    # Speech state
    current_speech = False
    speech_frames_left = 0
    frames_since_last_speech = 0
    
    # Main loop
    frame_count = 0
    
    while RUNNING:
        # Determine if this frame should have speech
        if speech_pattern == "random":
            # Random speech pattern
            if current_speech:
                # Continue existing speech if frames remain
                if speech_frames_left > 0:
                    speech_frames_left -= 1
                else:
                    current_speech = False
            else:
                # Randomly start new speech
                if random.random() < speech_frequency / frames_per_sec:
                    current_speech = True
                    speech_frames_left = int(speech_duration * frames_per_sec)
                    logger.info(f"Speech started at frame {frame_count}")
                
        elif speech_pattern == "periodic":
            # Periodic speech pattern
            frames_per_period = int(speech_period * frames_per_sec)
            speech_frames = int(speech_duration * frames_per_sec)
            
            if frames_since_last_speech >= frames_per_period:
                # Start new speech
                current_speech = True
                speech_frames_left = speech_frames
                frames_since_last_speech = 0
                logger.info(f"Speech started at frame {frame_count}")
            elif current_speech:
                # Continue existing speech if frames remain
                if speech_frames_left > 0:
                    speech_frames_left -= 1
                else:
                    current_speech = False
                    logger.info(f"Speech ended at frame {frame_count}")
            
            frames_since_last_speech += 1
        
        # Generate frame
        frame_data = generate_frame(
            sample_rate=sample_rate,
            frame_duration_ms=frame_duration_ms,
            is_speech=current_speech
        )
        
        frame_count += 1
        yield frame_data, current_speech
        
        # Add a small delay to simulate real-time processing
        time.sleep(frame_duration_ms / 1000)


def run_simulation(
    duration_sec=None,  # None for infinite
    output_dir="audiolog_sim_output",
    buffer_pre_sec=3.0,
    buffer_post_sec=3.0
):
    """
    Run the AudioLog simulation.
    
    Args:
        duration_sec: Duration in seconds (None for infinite)
        output_dir: Output directory for recordings
        buffer_pre_sec: Pre-buffer duration in seconds
        buffer_post_sec: Post-buffer duration in seconds
    """
    logger.info(f"Starting AudioLog simulation (buffer: pre={buffer_pre_sec}s, post={buffer_post_sec}s)")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample rate and other parameters
    sample_rate = 16000
    channels = 1
    frame_duration_ms = 30
    
    # Create components
    buffer = AudioBuffer(
        pre_buffer_seconds=buffer_pre_sec,
        post_buffer_seconds=buffer_post_sec,
        sample_rate=sample_rate,
        frame_duration_ms=frame_duration_ms
    )
    
    processor = AudioProcessor(
        audio_buffer=buffer,
        sample_rate=sample_rate,
        channels=channels
    )
    
    normalizer = AudioNormalizer(target_db=-16.0)
    encoder = AudioEncoder(sample_rate=sample_rate, channels=channels)
    storage = AudioStorage(output_dir)
    
    # Track state
    start_time = time.time()
    frame_count = 0
    segments_processed = 0
    
    # Create audio stream
    stream = simulate_audio_stream(
        sample_rate=sample_rate,
        frame_duration_ms=frame_duration_ms,
        speech_pattern="periodic",
        speech_period=7.0,      # Speech every 7 seconds
        speech_duration=2.0     # 2 seconds of speech
    )
    
    # Process frames
    try:
        for frame_data, is_speech in stream:
            # Check if simulation should end
            if duration_sec is not None and time.time() - start_time >= duration_sec:
                logger.info(f"Simulation duration reached ({duration_sec}s)")
                break
                
            # Add frame to buffer
            segment = buffer.add_frame(frame_data, is_speech)
            
            # Process segment if completed
            if segment is not None:
                # Convert to numpy for processing
                pcm_data = processor.frames_to_pcm(segment)
                audio_array = processor.pcm_to_numpy(pcm_data)
                
                # Normalize
                normalized = normalizer.normalize(audio_array)
                
                # Create temp file
                with tempfile.NamedTemporaryFile(suffix='.opus', delete=False) as tmp_file:
                    tmp_opus_path = tmp_file.name
                
                # Encode
                encoder.pcm_to_opus(normalized, tmp_opus_path)
                
                # Create metadata
                segment_duration = processor.get_duration_seconds(segment)
                now = time.time()
                start_time_ts = now - segment_duration
                
                metadata = {
                    'device_id': "sim_device",
                    'start_time': start_time_ts,
                    'end_time': now,
                    'duration': segment_duration,
                    'sample_rate': sample_rate,
                    'channels': channels,
                    'normalized': True,
                    'simulation': True
                }
                
                # Save to storage
                audio_path, metadata_path = storage.save_audio_file(
                    tmp_opus_path,
                    metadata,
                    device_id="sim_device",
                    timestamp=datetime.fromtimestamp(start_time_ts)
                )
                
                # Clean up temp file
                if os.path.exists(tmp_opus_path):
                    os.unlink(tmp_opus_path)
                
                # Update counters
                segments_processed += 1
                logger.info(f"Processed segment {segments_processed}: {audio_path} ({segment_duration:.2f}s)")
            
            frame_count += 1
            
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
    finally:
        # Final summary
        elapsed = time.time() - start_time
        logger.info(f"Simulation ended after {elapsed:.2f}s")
        logger.info(f"Processed {frame_count} frames and {segments_processed} segments")
        logger.info(f"Output files stored in: {output_dir}")


def signal_handler(sig, frame):
    """Handle Ctrl+C to stop simulation gracefully."""
    global RUNNING
    logger.info("Interrupt received, stopping simulation...")
    RUNNING = False


if __name__ == "__main__":
    import signal
    signal.signal(signal.SIGINT, signal_handler)
    
    # Run with a 30 second duration
    run_simulation(
        duration_sec=30,  # 30 second simulation
        buffer_pre_sec=2.0,
        buffer_post_sec=2.0
    )