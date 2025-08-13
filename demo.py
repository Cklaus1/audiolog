#!/usr/bin/env python3
"""
AudioLog Demo Script

This script demonstrates the core functionality of AudioLog without requiring 
microphone access or PyAudio, which may not be available in all environments.
"""

import os
import sys
import time
import logging
import argparse
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

def generate_synthetic_speech():
    """
    Generate synthetic speech waveform.
    
    Returns:
        Tuple of (pcm_data, start_time, end_time)
    """
    # Parameters
    sample_rate = 16000
    duration = 5.0  # 5 seconds
    speech_duration = 3.0  # 3 seconds of speech in the middle
    
    # Create time array
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Initialize empty audio
    audio = np.zeros(len(t), dtype=np.int16)
    
    # Add speech in the middle (simple sine wave)
    speech_start_idx = int((duration - speech_duration) / 2 * sample_rate)
    speech_end_idx = speech_start_idx + int(speech_duration * sample_rate)
    
    # Generate speech as a sine wave with amplitude modulation
    speech_t = t[speech_start_idx:speech_end_idx]
    carrier = np.sin(2 * np.pi * 440 * speech_t)  # 440 Hz carrier
    modulator = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * speech_t)  # 3 Hz modulation
    speech = (10000 * carrier * modulator).astype(np.int16)
    
    # Insert speech into audio
    audio[speech_start_idx:speech_end_idx] = speech
    
    # Convert to bytes
    pcm_data = audio.tobytes()
    
    # Set timestamps
    now = time.time()
    start_time = now - duration
    end_time = now
    
    return pcm_data, start_time, end_time


def demo_full_pipeline():
    """Demonstrate the full AudioLog pipeline with synthetic data."""
    logger.info("Starting AudioLog demo with synthetic speech...")
    
    # Generate synthetic speech
    pcm_data, start_time, end_time = generate_synthetic_speech()
    logger.info(f"Generated {len(pcm_data) / 2 / 16000:.2f} seconds of synthetic speech")
    
    # Create temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a custom config with the temp dir as storage
        config_path = os.path.join(temp_dir, "config.yaml")
        storage_path = os.path.join(temp_dir, "audiolog")
        os.makedirs(storage_path, exist_ok=True)
        
        # Convert to numpy for normalization
        audio_array = np.frombuffer(pcm_data, dtype=np.int16)
        
        logger.info("Normalizing audio...")
        normalizer = AudioNormalizer(target_db=-10.0)
        normalized = normalizer.normalize(audio_array)
        
        # Encode to opus
        logger.info("Encoding to Opus format...")
        encoder = AudioEncoder(sample_rate=16000, channels=1)
        opus_path = os.path.join(temp_dir, "speech.opus")
        encoder.pcm_to_opus(normalized, opus_path)
        
        # Create metadata
        metadata = {
            "device_id": "demo_device",
            "start_time": start_time,
            "end_time": end_time,
            "duration": end_time - start_time,
            "normalized": True,
            "demo": True
        }
        
        # Save to storage
        logger.info("Saving to storage with metadata...")
        storage = AudioStorage(storage_path)
        audio_path, metadata_path = storage.save_audio_file(
            opus_path,
            metadata,
            device_id="demo_device",
            timestamp=datetime.fromtimestamp(start_time)
        )
        
        logger.info(f"Saved audio file: {audio_path}")
        logger.info(f"Saved metadata: {metadata_path}")
        
        # List all files in storage
        logger.info("Listing files in storage...")
        files = storage.list_audio_files(include_metadata=True)
        
        for file_info in files:
            logger.info(f"File: {file_info['filename']}")
            if 'metadata' in file_info:
                logger.info(f"  Duration: {file_info['metadata']['duration']:.2f}s")
                logger.info(f"  Device: {file_info['metadata']['device_id']}")
                logger.info(f"  Timestamp: {datetime.fromtimestamp(file_info['metadata']['start_time']).isoformat()}")
                
        logger.info("Demo completed successfully!")
        logger.info(f"Files are stored in: {storage_path}")
        
        # Keep files for inspection if requested
        if args.keep_files:
            persistent_dir = os.path.join(os.getcwd(), "audiolog_demo_output")
            os.makedirs(persistent_dir, exist_ok=True)
            
            # Copy the files
            import shutil
            for root, dirs, files in os.walk(storage_path):
                for dir in dirs:
                    os.makedirs(os.path.join(persistent_dir, os.path.relpath(os.path.join(root, dir), storage_path)), exist_ok=True)
                    
                for file in files:
                    src = os.path.join(root, file)
                    dst = os.path.join(persistent_dir, os.path.relpath(src, storage_path))
                    shutil.copy2(src, dst)
            
            logger.info(f"Demo files copied to: {persistent_dir}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='AudioLog Demo'
    )
    
    parser.add_argument(
        '--keep-files',
        action='store_true',
        help='Keep generated files after demo'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    # Run demo
    demo_full_pipeline()