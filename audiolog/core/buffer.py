import time
import logging
import threading
import numpy as np
from collections import deque
from typing import List, Optional, Tuple, Deque, Dict, Any

from audiolog.utils.config import config

logger = logging.getLogger(__name__)

class AudioBuffer:
    """
    Audio buffer that maintains pre and post-speech audio segments.
    Implements a rolling buffer to store audio frames before voice is detected,
    and continues capturing for a configurable duration after voice stops.
    """
    
    def __init__(self,
                 pre_buffer_seconds: float = 120.0,  # 2 minutes by default
                 post_buffer_seconds: float = 120.0,  # 2 minutes by default
                 sample_rate: int = 16000,
                 frame_duration_ms: int = 30):
        """
        Initialize the audio buffer.
        
        Args:
            pre_buffer_seconds: Seconds of audio to keep before speech detection
            post_buffer_seconds: Seconds to continue recording after speech ends
            sample_rate: Audio sample rate in Hz
            frame_duration_ms: Frame duration in milliseconds
        """
        self.pre_buffer_seconds = pre_buffer_seconds
        self.post_buffer_seconds = post_buffer_seconds
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        
        # Calculate how many frames we need to store
        self.frames_per_second = 1000 / frame_duration_ms
        self.pre_buffer_frames = int(pre_buffer_seconds * self.frames_per_second)
        self.post_buffer_frames = int(post_buffer_seconds * self.frames_per_second)
        
        # Buffer to store audio frames
        self.buffer: Deque[bytes] = deque(maxlen=self.pre_buffer_frames)
        
        # State
        self.is_speech_active = False
        self.post_buffer_count = 0
        self.current_segment: List[bytes] = []
        self.last_speech_time = 0
        self.speech_start_time = 0
        
        # Thread safety
        self.lock = threading.RLock()
        
        logger.info(f"Initialized AudioBuffer (pre={pre_buffer_seconds}s, "
                   f"post={post_buffer_seconds}s, frames_per_second={self.frames_per_second})")
    
    def add_frame(self, frame: bytes, is_speech: bool) -> Optional[List[bytes]]:
        """
        Add an audio frame to the buffer and process based on speech detection.
        
        Args:
            frame: Raw audio frame data
            is_speech: Whether this frame contains speech
            
        Returns:
            List of frames if a complete segment is ready, None otherwise
        """
        with self.lock:
            now = time.time()
            
            # Always add the frame to the circular buffer
            self.buffer.append(frame)
            
            if is_speech:
                self.last_speech_time = now
                
                if not self.is_speech_active:
                    # Speech just started
                    self.is_speech_active = True
                    self.speech_start_time = now
                    self.current_segment = list(self.buffer)  # Copy pre-buffer
                    logger.debug(f"Speech started, captured {len(self.current_segment)} frames from pre-buffer")
                
                # Add the current frame to the current speech segment
                self.current_segment.append(frame)
                self.post_buffer_count = 0
            
            elif self.is_speech_active:
                # Speech was active, but current frame doesn't have speech
                self.current_segment.append(frame)
                self.post_buffer_count += 1
                
                # Check if we've reached the post-buffer limit
                if self.post_buffer_count >= self.post_buffer_frames:
                    self.is_speech_active = False
                    segment = self.current_segment
                    self.current_segment = []
                    logger.debug(f"Speech ended, captured {len(segment)} frames total "
                                f"(duration: {len(segment)/self.frames_per_second:.2f}s)")
                    return segment
            
            return None
    
    def reset(self) -> None:
        """Reset the buffer state."""
        with self.lock:
            self.buffer.clear()
            self.is_speech_active = False
            self.post_buffer_count = 0
            self.current_segment = []
            self.last_speech_time = 0
            self.speech_start_time = 0
            logger.debug("Reset audio buffer")
    
    def get_segment_info(self) -> Dict[str, Any]:
        """
        Get information about the current segment.
        
        Returns:
            Dictionary with segment details
        """
        with self.lock:
            return {
                "is_active": self.is_speech_active,
                "speech_start_time": self.speech_start_time,
                "last_speech_time": self.last_speech_time,
                "segment_frames": len(self.current_segment),
                "segment_duration": len(self.current_segment) / self.frames_per_second if self.current_segment else 0,
                "post_buffer_progress": self.post_buffer_count / self.post_buffer_frames if self.is_speech_active else 0
            }
    
    def force_complete_segment(self) -> Optional[List[bytes]]:
        """
        Force completion of the current segment.
        Useful when stopping recording or application shutdown.
        
        Returns:
            List of frames if a segment was in progress, None otherwise
        """
        with self.lock:
            if self.is_speech_active and self.current_segment:
                segment = self.current_segment
                self.is_speech_active = False
                self.current_segment = []
                self.post_buffer_count = 0
                return segment
            return None


class AudioProcessor:
    """
    Process audio segments from the buffer.
    """
    
    def __init__(self, 
                 audio_buffer: AudioBuffer,
                 sample_rate: int = 16000,
                 channels: int = 1,
                 sample_width: int = 2):  # 16-bit audio = 2 bytes
        """
        Initialize the audio processor.
        
        Args:
            audio_buffer: The AudioBuffer instance to get segments from
            sample_rate: Audio sample rate in Hz
            channels: Number of audio channels
            sample_width: Sample width in bytes
        """
        self.audio_buffer = audio_buffer
        self.sample_rate = sample_rate
        self.channels = channels
        self.sample_width = sample_width
        
    def frames_to_pcm(self, frames: List[bytes]) -> bytes:
        """
        Convert a list of audio frames to a single PCM audio block.
        
        Args:
            frames: List of audio frame bytes
            
        Returns:
            Combined PCM audio data
        """
        return b''.join(frames)
    
    def pcm_to_numpy(self, pcm_data: bytes) -> np.ndarray:
        """
        Convert PCM audio data to a numpy array.
        
        Args:
            pcm_data: Raw PCM audio data
            
        Returns:
            Numpy array of audio samples
        """
        # Convert to numpy array based on sample width
        if self.sample_width == 2:  # 16-bit
            return np.frombuffer(pcm_data, dtype=np.int16)
        elif self.sample_width == 4:  # 32-bit
            return np.frombuffer(pcm_data, dtype=np.int32)
        else:
            raise ValueError(f"Unsupported sample width: {self.sample_width}")
    
    def get_duration_seconds(self, frames: List[bytes]) -> float:
        """
        Calculate the duration of a list of frames in seconds.
        
        Args:
            frames: List of audio frame bytes
            
        Returns:
            Duration in seconds
        """
        total_samples = sum(len(frame) // self.sample_width for frame in frames)
        return total_samples / self.sample_rate