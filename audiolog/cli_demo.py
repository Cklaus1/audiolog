import os
import sys
import time
import json
import logging
import argparse
import threading
import signal
import tempfile
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

from audiolog.utils.config import config
from audiolog.core.buffer import AudioBuffer, AudioProcessor
from audiolog.audio.processor import AudioNormalizer, AudioEncoder
from audiolog.audio.storage import AudioStorage


# Global state for the application
class AudioLogDemoState:
    """Demo state for the AudioLog application (without hardware dependencies)."""
    
    def __init__(self):
        self.running = False
        self.paused = False
        self.buffer = None
        self.processor = None
        self.storage = None
        self.normalizer = None
        self.encoder = None
        self.segments_recorded = 0
        
    def initialize(self):
        """Initialize the state with components from configuration."""
        # Load config and print values
        sample_rate = config.get('audio.sample_rate')
        channels = config.get('audio.channels')
        pre_buffer_seconds = config.get('buffer.pre_buffer_seconds')
        post_buffer_seconds = config.get('buffer.post_buffer_seconds')

        # Debug log all config values
        logger.debug(f"Config values:")
        logger.debug(f"- buffer.pre_buffer_seconds: {pre_buffer_seconds}")
        logger.debug(f"- buffer.post_buffer_seconds: {post_buffer_seconds}")
        logger.debug(f"- audio.sample_rate: {sample_rate}")
        logger.debug(f"- audio.channels: {channels}")
        logger.debug(f"- storage.base_path: {config.get('storage.base_path')}")
        
        # Initialize components
        logger.debug(f"Initializing with pre_buffer={pre_buffer_seconds}s, post_buffer={post_buffer_seconds}s")
        self.buffer = AudioBuffer(
            pre_buffer_seconds=float(pre_buffer_seconds),
            post_buffer_seconds=float(post_buffer_seconds),
            sample_rate=sample_rate,
            frame_duration_ms=30
        )
        
        self.processor = AudioProcessor(
            audio_buffer=self.buffer,
            sample_rate=sample_rate,
            channels=channels
        )
        
        self.storage = AudioStorage()
        self.normalizer = AudioNormalizer()
        self.encoder = AudioEncoder(sample_rate=sample_rate, channels=channels)
        
        self.running = True
        self.paused = False
        self.segments_recorded = 0
        
        logger.info("AudioLog demo state initialized")
    
    def cleanup(self):
        """Clean up resources."""
        self.running = False
        logger.info("AudioLog demo state cleaned up")
    
    def process_synthetic_speech(self, duration=10.0, with_speech=True):
        """
        Process synthetic speech through the pipeline.
        
        Args:
            duration: Duration of synthetic audio in seconds
            with_speech: Whether to include speech
        """
        # Parameters
        sample_rate = config.get('audio.sample_rate')
        frame_duration_ms = 30
        samples_per_frame = int(sample_rate * frame_duration_ms / 1000)
        
        # Calculate number of frames
        frame_count = int(duration * 1000 / frame_duration_ms)
        
        # Process frames
        speech_active = False
        speech_frames = []
        speech_start_time = None
        
        logger.info(f"Generating {duration:.1f} seconds of synthetic audio ({frame_count} frames)")
        
        for i in range(frame_count):
            # Calculate time for this frame
            frame_time = i * frame_duration_ms / 1000
            
            # Determine if this frame should have speech
            # (speech in the middle third of the audio)
            is_speech = with_speech and (duration / 3 < frame_time < 2 * duration / 3)
            
            # Generate audio data
            if is_speech:
                # Generate sine wave (speech)
                t = np.linspace(0, frame_duration_ms/1000, samples_per_frame, endpoint=False)
                audio = (10000 * np.sin(2 * np.pi * 440 * t)).astype(np.int16)
                frame_data = audio.tobytes()
                
                # Log speech start
                if not speech_active:
                    speech_active = True
                    speech_start_time = time.time()
                    logger.info(f"Speech started at frame {i} (time: {frame_time:.2f}s)")
            else:
                # Generate silence
                frame_data = np.zeros(samples_per_frame, dtype=np.int16).tobytes()
                
                # Log speech end
                if speech_active:
                    speech_active = False
                    speech_duration = time.time() - speech_start_time
                    logger.info(f"Speech ended at frame {i} (time: {frame_time:.2f}s, duration: {speech_duration:.2f}s)")
            
            # Add to buffer
            if not self.paused:
                segment = self.buffer.add_frame(frame_data, is_speech)
                
                # Process completed segment
                if segment:
                    self.process_segment(segment)
            
            # Simulate processing time
            time.sleep(0.01)
            
            # Stop if requested
            if not self.running:
                break
    
    def process_segment(self, segment: List[bytes]) -> None:
        """
        Process an audio segment.
        
        Args:
            segment: List of audio frames
        """
        if self.paused:
            logger.debug("Recording paused, skipping segment processing")
            return
            
        try:
            # Get raw PCM data
            pcm_data = self.processor.frames_to_pcm(segment)
            
            # Convert to numpy for normalization
            audio_array = self.processor.pcm_to_numpy(pcm_data)
            
            # Normalize the audio
            normalized = self.normalizer.normalize(audio_array)
            
            # Get duration
            duration = self.processor.get_duration_seconds(segment)
            
            # Create a temporary file for the opus data
            with tempfile.NamedTemporaryFile(suffix='.opus', delete=False) as tmp_file:
                tmp_opus_path = tmp_file.name
            
            # Encode to opus
            self.encoder.pcm_to_opus(
                normalized,
                output_path=tmp_opus_path
            )
            
            # Create metadata
            now = time.time()
            start_time = now - duration
            device_id = config.get('device.id')
            
            metadata = {
                'device_id': device_id,
                'start_time': start_time,
                'end_time': now,
                'duration': duration,
                'sample_rate': self.processor.sample_rate,
                'channels': self.processor.channels,
                'normalized': True
            }
            
            # Save to storage
            opus_path, metadata_path = self.storage.save_audio_file(
                tmp_opus_path,
                metadata,
                device_id=device_id,
                timestamp=datetime.fromtimestamp(start_time)
            )
            
            # Clean up temp file
            if os.path.exists(tmp_opus_path):
                os.unlink(tmp_opus_path)
                
            # Update stats
            self.segments_recorded += 1
            
            logger.info(f"Processed segment: {opus_path} ({duration:.2f}s)")
            
        except Exception as e:
            logger.error(f"Error processing segment: {e}")


# Create global state instance
state = AudioLogDemoState()

# Set up logging
logger = logging.getLogger("audiolog")


def setup_logging(level: int = logging.INFO, log_file: Optional[str] = None):
    """
    Set up logging configuration.
    
    Args:
        level: Logging level
        log_file: Optional log file path
    """
    handlers = [logging.StreamHandler()]
    
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def signal_handler(sig, frame):
    """Handle signals for graceful shutdown."""
    logger.info("Received signal to exit")
    state.running = False


def run_demo(duration: float = 60.0):
    """Run a demo of AudioLog with synthetic data."""
    try:
        # Initialize state
        state.initialize()
        
        # Set up signal handler
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Process synthetic speech
        logger.info(f"Starting demo with {duration:.1f} seconds of synthetic audio")
        state.process_synthetic_speech(duration=duration)
        
        logger.info(f"Demo complete. Recorded {state.segments_recorded} segments")
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error in demo: {e}")
    finally:
        # Clean up
        state.cleanup()
        logger.info("Demo stopped")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='AudioLog Demo - continuous audio recording with voice detection'
    )
    
    # Demo options
    parser.add_argument(
        '-d', '--duration',
        type=float,
        default=30.0,
        help='Duration of demo in seconds (default: 30)'
    )
    
    # Config options
    parser.add_argument(
        '-c', '--config',
        help='Path to config file'
    )
    
    # Logging options
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--log-file',
        help='Path to log file'
    )
    
    # Other options
    parser.add_argument(
        '--version',
        action='store_true',
        help='Show version and exit'
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the AudioLog application."""
    # Parse command line arguments
    args = parse_args()
    
    # Show version if requested
    if args.version:
        print("AudioLog v0.1.0 (Demo)")
        return 0
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level, log_file=args.log_file)
    
    try:
        # Load custom config if provided
        if args.config:
            config_path = os.path.abspath(args.config)
            if not os.path.exists(config_path):
                logger.error(f"Config file not found: {config_path}")
                return 1
                
            config._instance = None  # Reset singleton
            config.__init__(config_path)
            logger.info(f"Loaded config from {config_path}")
        
        # Run demo
        run_demo(duration=args.duration)
            
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())