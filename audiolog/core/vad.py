import os
import time
import wave
import logging
import webrtcvad
import pyaudio
import numpy as np
from typing import Optional, Generator, Tuple, List, BinaryIO
from pathlib import Path
from datetime import datetime

from audiolog.utils.config import config

logger = logging.getLogger(__name__)

class VoiceDetector:
    """Voice Activity Detector using WebRTC VAD."""
    
    def __init__(self, 
                 aggressiveness: int = 3, 
                 sample_rate: int = 16000,
                 frame_duration_ms: int = 30):
        """
        Initialize the voice detector.
        
        Args:
            aggressiveness: VAD aggressiveness mode (0-3), higher is more aggressive
            sample_rate: Audio sample rate in Hz
            frame_duration_ms: Frame duration in milliseconds
        """
        self.vad = webrtcvad.Vad(aggressiveness)
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)  # samples per frame
        self.voice_detected = False
        self.last_detection_time = 0
        
        # Validation
        if self.sample_rate not in [8000, 16000, 32000, 48000]:
            raise ValueError("Sample rate must be 8000, 16000, 32000, or 48000 Hz")
        
        if self.frame_duration_ms not in [10, 20, 30]:
            raise ValueError("Frame duration must be 10, 20, or 30 ms")
            
        logger.info(f"Initialized VAD (aggressiveness={aggressiveness}, "
                   f"sample_rate={sample_rate}Hz, frame_duration={frame_duration_ms}ms)")
    
    def is_speech(self, audio_frame: bytes) -> bool:
        """
        Detect speech in a single frame of audio.
        
        Args:
            audio_frame: Raw audio data (16-bit PCM, mono)
            
        Returns:
            True if speech is detected, False otherwise
        """
        try:
            if len(audio_frame) != self.frame_size * 2:  # 16-bit = 2 bytes per sample
                logger.warning(f"Frame size mismatch: expected {self.frame_size * 2} bytes, "
                              f"got {len(audio_frame)} bytes")
                return False
                
            result = self.vad.is_speech(audio_frame, self.sample_rate)
            
            # Update detection state
            now = time.time()
            if result:
                if not self.voice_detected:
                    logger.info(f"Voice detected at {datetime.now().isoformat()}")
                self.voice_detected = True
                self.last_detection_time = now
            else:
                self.voice_detected = False
                
            return result
        except Exception as e:
            logger.error(f"Error in VAD processing: {e}")
            return False
    
    def process_audio_stream(self, 
                             audio_stream: Generator[bytes, None, None], 
                             callback=None) -> Generator[Tuple[bytes, bool], None, None]:
        """
        Process a stream of audio frames and detect speech.
        
        Args:
            audio_stream: Generator yielding audio frames
            callback: Optional callback function(frame, is_speech)
            
        Yields:
            Tuple of (audio_frame, is_speech)
        """
        for frame in audio_stream:
            is_speech = self.is_speech(frame)
            
            if callback:
                callback(frame, is_speech)
                
            yield frame, is_speech


class MicrophoneAudioStream:
    """Stream audio from microphone."""
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 channels: int = 1,
                 chunk_size: int = 480,
                 format: int = pyaudio.paInt16):
        """
        Initialize the microphone stream.
        
        Args:
            sample_rate: Audio sample rate in Hz
            channels: Number of audio channels (1 for mono)
            chunk_size: Samples per buffer
            format: PyAudio format
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.format = format
        self.pyaudio = pyaudio.PyAudio()
        self.stream = None
        self.active = False
        
    def __enter__(self):
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        
    def start(self):
        """Start the microphone stream."""
        logger.info(f"Starting microphone stream (rate={self.sample_rate}Hz, "
                   f"channels={self.channels}, chunk_size={self.chunk_size})")
        
        self.stream = self.pyaudio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        self.active = True
        return self
        
    def stop(self):
        """Stop the microphone stream."""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        self.active = False
        logger.info("Stopped microphone stream")
        
    def close(self):
        """Close PyAudio."""
        self.stop()
        self.pyaudio.terminate()
        logger.info("Closed PyAudio instance")
        
    def read(self) -> bytes:
        """Read a chunk of audio data."""
        if not self.active or not self.stream:
            raise RuntimeError("Stream is not active")
            
        return self.stream.read(self.chunk_size, exception_on_overflow=False)
        
    def get_stream(self) -> Generator[bytes, None, None]:
        """Get a generator yielding audio chunks."""
        while self.active:
            yield self.read()


class FileAudioStream:
    """Stream audio from a file."""
    
    def __init__(self, 
                 file_path: str,
                 chunk_size: int = 480):
        """
        Initialize file audio stream.
        
        Args:
            file_path: Path to the audio file
            chunk_size: Samples per buffer
        """
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.wave_file = None
        self.sample_rate = None
        self.channels = None
        self.active = False
        
        # Validate file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
            
    def __enter__(self):
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        
    def start(self):
        """Open the audio file for reading."""
        logger.info(f"Opening audio file: {self.file_path}")
        
        self.wave_file = wave.open(self.file_path, 'rb')
        self.sample_rate = self.wave_file.getframerate()
        self.channels = self.wave_file.getnchannels()
        self.active = True
        
        logger.info(f"File details: rate={self.sample_rate}Hz, "
                   f"channels={self.channels}, chunk_size={self.chunk_size}")
        return self
        
    def stop(self):
        """Close the audio file."""
        if self.wave_file:
            self.wave_file.close()
            self.wave_file = None
        
        self.active = False
        logger.info(f"Closed audio file: {self.file_path}")
        
    def read(self) -> bytes:
        """Read a chunk of audio data."""
        if not self.active or not self.wave_file:
            raise RuntimeError("File stream is not active")
            
        data = self.wave_file.readframes(self.chunk_size)
        if not data:
            self.active = False
            
        return data
        
    def get_stream(self) -> Generator[bytes, None, None]:
        """Get a generator yielding audio chunks."""
        while self.active:
            data = self.read()
            if not data:
                break
            yield data