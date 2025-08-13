import os
import sys
import time
import json
import logging
import argparse
import threading
import signal
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

from audiolog.utils.config import config
from audiolog.core.vad import VoiceDetector, MicrophoneAudioStream, FileAudioStream
from audiolog.core.buffer import AudioBuffer, AudioProcessor
from audiolog.audio.processor import AudioNormalizer, AudioEncoder
from audiolog.audio.storage import AudioStorage
from audiolog.ui.tray import create_tray_controller
from audiolog.sync.drive import GoogleDriveSync
from audiolog.transcription.whisper import TranscriptionManager

logger = logging.getLogger(__name__)

# Global state for the application
class AudioLogState:
    """Global state for the AudioLog application."""
    
    def __init__(self):
        self.running = False
        self.paused = False
        self.audio_stream = None
        self.vad = None
        self.buffer = None
        self.processor = None
        self.storage = None
        self.normalizer = None
        self.encoder = None
        self.drive_sync = None
        self.transcription = None
        self.tray_controller = None
        self.segments_recorded = 0
        
    def initialize(self):
        """Initialize the state with components from configuration."""
        # Load config
        sample_rate = config.get('audio.sample_rate')
        channels = config.get('audio.channels')
        chunk_size = config.get('audio.chunk_size')
        pre_buffer_seconds = config.get('buffer.pre_buffer_seconds')
        post_buffer_seconds = config.get('buffer.post_buffer_seconds')
        
        # Initialize components
        self.vad = VoiceDetector(
            aggressiveness=3,
            sample_rate=sample_rate,
            frame_duration_ms=30
        )
        
        self.buffer = AudioBuffer(
            pre_buffer_seconds=pre_buffer_seconds,
            post_buffer_seconds=post_buffer_seconds,
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
        
        # Initialize sync if enabled
        if config.get('google_drive.enabled', False):
            self.drive_sync = GoogleDriveSync()
            self.drive_sync.start_sync_thread()
        
        # Initialize transcription if enabled
        if config.get('transcription.enabled', True):
            self.transcription = TranscriptionManager(
                model_name=config.get('transcription.model', 'tiny'),
                format=config.get('transcription.format', 'json')
            )
            self.transcription.start_watching()
        
        # Initialize UI if enabled
        if config.get('ui.enabled', True):
            self.tray_controller = create_tray_controller(
                recording_status_callback=lambda: not self.paused,
                toggle_callback=self.toggle_pause
            )
        
        self.running = True
        self.paused = False
        self.segments_recorded = 0
        
        logger.info("AudioLog state initialized")
    
    def cleanup(self):
        """Clean up resources."""
        self.running = False
        
        # Stop components
        if self.audio_stream:
            self.audio_stream.stop()
            if hasattr(self.audio_stream, 'close'):
                self.audio_stream.close()
        
        # Stop drive sync
        if self.drive_sync:
            self.drive_sync.stop_sync_thread()
            
        # Stop transcription
        if self.transcription:
            self.transcription.stop_watching()
            
        logger.info("AudioLog state cleaned up")
    
    def toggle_pause(self, recording_active: bool) -> None:
        """
        Toggle recording pause state.
        
        Args:
            recording_active: Whether recording should be active
        """
        self.paused = not recording_active
        logger.info(f"Recording {'resumed' if not self.paused else 'paused'}")
        
        # Force complete current segment if pausing
        if self.paused and self.buffer:
            segment = self.buffer.force_complete_segment()
            if segment:
                self.process_segment(segment)
    
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


# Global state instance
state = AudioLogState()


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


def process_audio_stream():
    """Process the audio stream and detect voice."""
    if not state.audio_stream or not state.vad or not state.buffer:
        logger.error("Audio components not initialized")
        return
    
    try:
        # Get audio stream
        audio_stream = state.audio_stream.get_stream()
        
        # Process audio frames
        for frame in audio_stream:
            if not state.running:
                break
                
            if state.paused:
                continue
                
            # Detect speech
            is_speech = state.vad.is_speech(frame)
            
            # Add to buffer
            segment = state.buffer.add_frame(frame, is_speech)
            
            # If we have a complete segment, process it
            if segment:
                state.process_segment(segment)
    
    except Exception as e:
        logger.error(f"Error processing audio stream: {e}")
        state.running = False


def signal_handler(sig, frame):
    """Handle signals for graceful shutdown."""
    logger.info("Received signal to exit")
    state.running = False


def run_with_mic():
    """Run AudioLog with microphone input."""
    try:
        # Initialize state
        state.initialize()
        
        # Create and start microphone stream
        state.audio_stream = MicrophoneAudioStream(
            sample_rate=config.get('audio.sample_rate'),
            channels=config.get('audio.channels'),
            chunk_size=config.get('audio.chunk_size')
        ).start()
        
        # Set up signal handler
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Process audio in the main thread
        logger.info("Starting audio processing")
        process_audio_stream()
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        # Clean up
        state.cleanup()
        logger.info("AudioLog stopped")


def run_with_file(file_path: str):
    """
    Run AudioLog with a file input for testing.
    
    Args:
        file_path: Path to the audio file
    """
    try:
        # Initialize state
        state.initialize()
        
        # Create and start file stream
        state.audio_stream = FileAudioStream(
            file_path=file_path,
            chunk_size=config.get('audio.chunk_size')
        ).start()
        
        # Set up signal handler
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Process audio in the main thread
        logger.info(f"Processing file: {file_path}")
        process_audio_stream()
        
        logger.info(f"File processing complete. Recorded {state.segments_recorded} segments")
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        # Clean up
        state.cleanup()
        logger.info("AudioLog stopped")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='AudioLog - continuous audio recording with voice detection'
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        '-f', '--file',
        help='Process an audio file (for testing)'
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
    
    # UI options
    parser.add_argument(
        '--no-ui',
        action='store_true',
        help='Disable system tray UI'
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
        print("AudioLog v0.1.0")
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
        
        # Override config with command line arguments
        if args.no_ui:
            config.set('ui.enabled', False)
        
        # Run with file or microphone
        if args.file:
            if not os.path.exists(args.file):
                logger.error(f"Audio file not found: {args.file}")
                return 1
                
            run_with_file(args.file)
        else:
            run_with_mic()
            
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())