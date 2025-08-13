import os
import time
import logging
import tempfile
import numpy as np
import ffmpeg
from typing import List, Optional, Tuple, Dict, Any, Union
from pathlib import Path

from audiolog.utils.config import config

logger = logging.getLogger(__name__)

class AudioNormalizer:
    """Normalize audio volume to a consistent level."""
    
    def __init__(self, target_db: float = -16.0):
        """
        Initialize the audio normalizer.
        
        Args:
            target_db: Target dBFS level for normalization
        """
        self.target_db = target_db
        logger.debug(f"Initialized AudioNormalizer (target_db={target_db})")
    
    def get_db_level(self, audio: np.ndarray) -> float:
        """
        Calculate the RMS level of audio in dBFS.

        Args:
            audio: Audio samples as numpy array

        Returns:
            RMS level in dBFS
        """
        if len(audio) == 0:
            return -np.inf

        # Calculate RMS
        rms = np.sqrt(np.mean(np.square(audio.astype(np.float32))))

        # Convert to dBFS (full scale)
        if rms > 0:
            # Get max value based on data type
            if audio.dtype == np.int16:
                max_value = 32767
            elif audio.dtype == np.int32:
                max_value = 2147483647
            elif audio.dtype == np.float32:
                max_value = 1.0
            else:
                max_value = np.iinfo(np.int16).max  # Default to int16

            db = 20 * np.log10(rms / max_value)
        else:
            db = -np.inf

        return db
    
    def normalize(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to target dBFS level.
        
        Args:
            audio: Audio samples as numpy array
            
        Returns:
            Normalized audio samples
        """
        if len(audio) == 0:
            return audio
            
        current_db = self.get_db_level(audio)
        logger.debug(f"Current audio level: {current_db:.2f} dBFS, target: {self.target_db:.2f} dBFS")
        
        # Skip if audio is silence
        if current_db <= -80:
            logger.debug("Audio is too quiet, skipping normalization")
            return audio
            
        # Calculate gain
        gain_db = self.target_db - current_db
        gain_linear = 10 ** (gain_db / 20.0)
        
        # Apply gain
        normalized = audio.astype(np.float32) * gain_linear
        
        # Clip to avoid distortion
        normalized = np.clip(normalized, 
                             np.iinfo(audio.dtype).min, 
                             np.iinfo(audio.dtype).max)
        
        logger.debug(f"Applied {gain_db:.2f} dB gain, result: {self.get_db_level(normalized):.2f} dBFS")
        return normalized.astype(audio.dtype)


class AudioEncoder:
    """Encode audio to various formats, primarily Opus."""
    
    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        """
        Initialize the audio encoder.
        
        Args:
            sample_rate: Audio sample rate in Hz
            channels: Number of audio channels
        """
        self.sample_rate = sample_rate
        self.channels = channels
        logger.debug(f"Initialized AudioEncoder (rate={sample_rate}Hz, channels={channels})")
    
    def pcm_to_opus(self, 
                    pcm_data: Union[bytes, np.ndarray], 
                    output_path: Optional[str] = None,
                    bitrate: str = "32k") -> str:
        """
        Convert raw PCM audio to Opus format.
        
        Args:
            pcm_data: Raw PCM audio data
            output_path: Output file path (optional)
            bitrate: Opus bitrate
            
        Returns:
            Path to the encoded Opus file
        """
        # Create temporary file if output_path not provided
        if output_path is None:
            tmp_dir = tempfile.gettempdir()
            output_path = os.path.join(tmp_dir, f"audiolog_{int(time.time())}.opus")
            
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Create temporary PCM file to use as input
        with tempfile.NamedTemporaryFile(suffix='.pcm', delete=False) as tmp_file:
            tmp_pcm_path = tmp_file.name
            
            if isinstance(pcm_data, np.ndarray):
                if pcm_data.dtype == np.int16:
                    tmp_file.write(pcm_data.tobytes())
                else:
                    tmp_file.write(pcm_data.astype(np.int16).tobytes())
            else:
                tmp_file.write(pcm_data)
        
        try:
            # Use ffmpeg to convert PCM to Opus
            logger.debug(f"Encoding audio to Opus format: {output_path}")
            
            ffmpeg.input(
                tmp_pcm_path,
                format='s16le',  # signed 16-bit little-endian
                channels=self.channels,
                sample_rate=self.sample_rate
            ).output(
                output_path,
                acodec='libopus',
                audio_bitrate=bitrate
            ).run(quiet=True, overwrite_output=True)
            
            logger.info(f"Successfully encoded audio to Opus: {output_path}")
            return output_path
            
        except ffmpeg.Error as e:
            logger.error(f"Failed to encode audio to Opus: {e.stderr.decode() if e.stderr else str(e)}")
            raise
        finally:
            # Clean up the temporary PCM file
            if os.path.exists(tmp_pcm_path):
                os.unlink(tmp_pcm_path)
    
    def process_audio(self, 
                      pcm_data: Union[bytes, np.ndarray], 
                      output_path: str,
                      normalize: bool = True,
                      target_db: float = -16.0,
                      bitrate: str = "32k") -> str:
        """
        Process audio data: normalize and encode to Opus.
        
        Args:
            pcm_data: Raw PCM audio data
            output_path: Output file path
            normalize: Whether to normalize audio
            target_db: Target dBFS level for normalization
            bitrate: Opus bitrate
            
        Returns:
            Path to the processed audio file
        """
        # Convert bytes to numpy array if needed
        if isinstance(pcm_data, bytes):
            audio_array = np.frombuffer(pcm_data, dtype=np.int16)
        else:
            audio_array = pcm_data
            
        # Normalize audio if requested
        if normalize:
            normalizer = AudioNormalizer(target_db=target_db)
            audio_array = normalizer.normalize(audio_array)
            
        # Encode to Opus
        return self.pcm_to_opus(audio_array, output_path, bitrate=bitrate)