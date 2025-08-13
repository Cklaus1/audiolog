#!/usr/bin/env python3
"""
Headless AudioLog Runner

This script runs AudioLog in a headless environment without requiring microphone
input or GUI components. It uses simulated audio input instead.
"""

import os
import sys
import time
import signal
import logging
import argparse
from pathlib import Path

# Add main directory to import path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import AudioLog components
from audiolog.utils.config import config
from audiolog.core.buffer import AudioBuffer, AudioProcessor
from audiolog.audio.processor import AudioNormalizer, AudioEncoder
from audiolog.audio.storage import AudioStorage
from audiolog.transcription.whisper import TranscriptionManager
from simulate_recording import generate_frame, RUNNING

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("headless-audiolog")

def setup_signal_handlers():
    """Set up signal handlers for graceful shutdown."""
    def signal_handler(sig, frame):
        global RUNNING
        logger.info("Interrupt received, stopping AudioLog...")
        RUNNING = False
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Headless AudioLog - runs without microphone or UI'
    )
    
    parser.add_argument(
        '-c', '--config',
        default='/shared/projects/audiolog/configs/headless_config.yaml',
        help='Path to config file'
    )
    
    parser.add_argument(
        '-d', '--duration',
        type=int,
        default=None,
        help='Duration to run in seconds (default: run indefinitely)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--speech-pattern',
        choices=['random', 'periodic'],
        default='periodic',
        help='Pattern for simulated speech (default: periodic)'
    )
    
    parser.add_argument(
        '--speech-period',
        type=float,
        default=7.0,
        help='Seconds between speech events for periodic pattern (default: 7.0)'
    )
    
    parser.add_argument(
        '--speech-duration',
        type=float,
        default=2.0,
        help='Duration of speech events in seconds (default: 2.0)'
    )
    
    return parser.parse_args()

def simulate_audio_stream(
    sample_rate=16000,
    frame_duration_ms=30,
    speech_pattern="periodic",
    speech_frequency=0.2,
    speech_period=7.0,
    speech_duration=2.0
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

def run_headless_audiolog(args):
    """
    Run AudioLog in headless mode with simulated audio input.
    
    Args:
        args: Command line arguments
    """
    global RUNNING
    
    # Load configuration
    if args.config:
        config_path = os.path.abspath(args.config)
        if not os.path.exists(config_path):
            logger.error(f"Config file not found: {config_path}")
            return 1
            
        config._instance = None  # Reset singleton
        config.__init__(config_path)
        logger.info(f"Loaded config from {config_path}")
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create output directory
    output_dir = config.get('storage.base_path')
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample rate and other parameters
    sample_rate = config.get('audio.sample_rate')
    channels = config.get('audio.channels')
    frame_duration_ms = 30  # WebRTC VAD requires 10, 20, or 30 ms
    
    # Create components
    buffer = AudioBuffer(
        pre_buffer_seconds=config.get('buffer.pre_buffer_seconds'),
        post_buffer_seconds=config.get('buffer.post_buffer_seconds'),
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
    
    # Initialize transcription if enabled
    transcription = None
    if config.get('transcription.enabled', False):
        transcription = TranscriptionManager(
            model_name=config.get('transcription.model', 'tiny'),
            format=config.get('transcription.format', 'json')
        )
        transcription.start_watching()
    
    # Track state
    start_time = time.time()
    frame_count = 0
    segments_processed = 0
    
    # Create audio stream
    stream = simulate_audio_stream(
        sample_rate=sample_rate,
        frame_duration_ms=frame_duration_ms,
        speech_pattern=args.speech_pattern,
        speech_period=args.speech_period,
        speech_duration=args.speech_duration
    )
    
    logger.info(f"Starting headless AudioLog with simulated audio input")
    logger.info(f"Buffer: pre={config.get('buffer.pre_buffer_seconds')}s, "
               f"post={config.get('buffer.post_buffer_seconds')}s")
    logger.info(f"Speech pattern: {args.speech_pattern}")
    
    if args.duration:
        logger.info(f"Will run for {args.duration} seconds")
    else:
        logger.info("Will run until interrupted (press Ctrl+C to stop)")
    
    try:
        # Process frames
        for frame_data, is_speech in stream:
            # Check if we've reached the time limit
            if args.duration and time.time() - start_time >= args.duration:
                logger.info(f"Duration limit reached ({args.duration}s)")
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
                
                # Create temp file for Opus encoding
                with tempfile.NamedTemporaryFile(suffix='.opus', delete=False) as tmp_file:
                    tmp_opus_path = tmp_file.name
                
                # Encode
                encoder.pcm_to_opus(normalized, tmp_opus_path)
                
                # Create metadata
                segment_duration = processor.get_duration_seconds(segment)
                now = time.time()
                start_time_ts = now - segment_duration
                
                metadata = {
                    'device_id': config.get('device.id'),
                    'start_time': start_time_ts,
                    'end_time': now,
                    'duration': segment_duration,
                    'sample_rate': sample_rate,
                    'channels': channels,
                    'normalized': True,
                    'headless': True
                }
                
                # Save to storage
                audio_path, metadata_path = storage.save_audio_file(
                    tmp_opus_path,
                    metadata,
                    device_id=config.get('device.id'),
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
        logger.info("Interrupted by user")
    finally:
        # Clean up
        if transcription:
            transcription.stop_watching()
            
        # Final summary
        elapsed = time.time() - start_time
        logger.info(f"AudioLog stopped after {elapsed:.2f}s")
        logger.info(f"Processed {frame_count} frames and {segments_processed} segments")
        logger.info(f"Output files stored in: {os.path.expanduser(output_dir)}")
        
    return 0

if __name__ == "__main__":
    # Fix imports
    import random
    import tempfile
    from datetime import datetime
    
    # Set up signal handlers
    setup_signal_handlers()
    
    # Parse arguments
    args = parse_args()
    
    # Run AudioLog
    sys.exit(run_headless_audiolog(args))