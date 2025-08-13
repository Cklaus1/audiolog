import os
import json
import time
import logging
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
import whisper
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from audiolog.utils.config import config
from audiolog.audio.storage import AudioStorage

logger = logging.getLogger(__name__)

class AudioFileHandler(FileSystemEventHandler):
    """
    Watchdog handler for new audio files.
    Triggers transcription when new opus files are created.
    """
    
    def __init__(self, transcription_manager):
        """
        Initialize the file handler.
        
        Args:
            transcription_manager: TranscriptionManager instance
        """
        self.transcription_manager = transcription_manager
        
    def on_created(self, event):
        """
        Handle file creation events.
        
        Args:
            event: File event
        """
        if not event.is_directory and event.src_path.endswith('.opus'):
            logger.debug(f"New audio file detected: {event.src_path}")
            self.transcription_manager.queue_transcription(event.src_path)
            
    def on_modified(self, event):
        """
        Handle file modification events.
        
        Args:
            event: File event
        """
        # Only queue non-directory .opus files that don't already have a transcript
        if (not event.is_directory and event.src_path.endswith('.opus') and
                not self._has_transcript(event.src_path)):
            logger.debug(f"Modified audio file detected: {event.src_path}")
            self.transcription_manager.queue_transcription(event.src_path)
    
    def _has_transcript(self, audio_path: str) -> bool:
        """
        Check if a transcript already exists for an audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            True if a transcript exists, False otherwise
        """
        transcript_json = f"{audio_path}.transcript.json"
        transcript_txt = f"{audio_path}.transcript.txt"
        return os.path.exists(transcript_json) or os.path.exists(transcript_txt)


class TranscriptionManager:
    """
    Manages audio transcription using Whisper.
    Watches for new files and runs transcription in a background thread.
    """
    
    def __init__(self, 
                 base_path: Optional[str] = None,
                 model_name: str = "tiny",
                 format: str = "json",
                 language: Optional[str] = None,
                 on_transcription_complete: Optional[Callable[[str, Dict[str, Any]], None]] = None):
        """
        Initialize the transcription manager.
        
        Args:
            base_path: Base directory to watch for audio files
            model_name: Whisper model name (tiny, base, small, medium, large)
            format: Output format (json or txt)
            language: Language code (default: auto-detect)
            on_transcription_complete: Callback when transcription is done
        """
        if base_path is None:
            base_path = config.get('storage.base_path')
            
        # Expand user path if needed
        if '~' in base_path:
            base_path = os.path.expanduser(base_path)
            
        self.base_path = Path(base_path)
        self.model_name = model_name
        self.format = format
        self.language = language
        self.on_transcription_complete = on_transcription_complete
        
        # State
        self.model = None
        self.queue = []
        self.queue_lock = threading.RLock()
        self.is_transcribing = False
        self.observer = None
        self.thread = None
        self.running = False
        
        logger.debug(f"Initialized TranscriptionManager (base_path={base_path}, model={model_name})")
    
    def load_model(self) -> None:
        """Load the Whisper model."""
        logger.info(f"Loading Whisper model: {self.model_name}")
        self.model = whisper.load_model(self.model_name)
        logger.info(f"Whisper model {self.model_name} loaded successfully")
    
    def start_watching(self) -> None:
        """Start watching for new audio files."""
        if not config.get('transcription.enabled', True):
            logger.info("Transcription is disabled in configuration")
            return
            
        if self.observer:
            logger.warning("Already watching for files")
            return
            
        # Create the base directory if it doesn't exist
        if not self.base_path.exists():
            self.base_path.mkdir(parents=True, exist_ok=True)
            
        # Create and start the observer
        self.observer = Observer()
        self.observer.schedule(
            AudioFileHandler(self),
            str(self.base_path),
            recursive=True
        )
        self.observer.start()
        
        # Start the transcription thread
        self.running = True
        self.thread = threading.Thread(
            target=self._transcription_worker,
            daemon=True
        )
        self.thread.start()
        
        logger.info(f"Started watching for audio files in {self.base_path}")
    
    def stop_watching(self) -> None:
        """Stop watching for new audio files."""
        self.running = False
        
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None
            
        if self.thread:
            self.thread.join(timeout=5.0)
            self.thread = None
            
        logger.info("Stopped watching for audio files")
    
    def queue_transcription(self, audio_path: str) -> None:
        """
        Queue an audio file for transcription.
        
        Args:
            audio_path: Path to the audio file
        """
        with self.queue_lock:
            if audio_path not in self.queue:
                self.queue.append(audio_path)
                logger.debug(f"Queued for transcription: {audio_path}")
    
    def _transcription_worker(self) -> None:
        """Worker function for the transcription thread."""
        while self.running:
            try:
                # Process any queued files
                audio_path = None
                with self.queue_lock:
                    if self.queue:
                        audio_path = self.queue.pop(0)
                
                if audio_path:
                    # Load model if needed
                    if self.model is None:
                        self.load_model()
                        
                    # Transcribe the file
                    self.transcribe_file(audio_path)
                else:
                    # No files to process, sleep for a bit
                    time.sleep(1.0)
                    
            except Exception as e:
                logger.error(f"Error in transcription worker: {e}")
                time.sleep(5.0)
    
    def transcribe_file(self, audio_path: str) -> Optional[Dict[str, Any]]:
        """
        Transcribe an audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Transcription result or None if failed
        """
        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            return None
            
        try:
            self.is_transcribing = True
            logger.info(f"Transcribing: {audio_path}")
            
            # Load model if needed
            if self.model is None:
                self.load_model()
            
            # Transcribe the audio
            transcription_options = {
                'verbose': False,
                'fp16': False  # Use fp32 for better compatibility
            }
            
            if self.language:
                transcription_options['language'] = self.language
                
            # Run transcription
            result = self.model.transcribe(audio_path, **transcription_options)
            
            # Save the result
            transcript_path = self._save_transcript(audio_path, result)
            
            # Call the callback if provided
            if self.on_transcription_complete:
                self.on_transcription_complete(audio_path, result)
                
            logger.info(f"Transcription complete: {transcript_path}")
            return result
            
        except Exception as e:
            logger.error(f"Transcription failed for {audio_path}: {e}")
            return None
        finally:
            self.is_transcribing = False
    
    def _save_transcript(self, audio_path: str, result: Dict[str, Any]) -> str:
        """
        Save the transcription result.
        
        Args:
            audio_path: Path to the audio file
            result: Transcription result
            
        Returns:
            Path to the saved transcript
        """
        format = self.format.lower()
        
        # Generate the transcript path
        if format == 'json':
            transcript_path = f"{audio_path}.transcript.json"
            
            # Save as JSON
            with open(transcript_path, 'w') as f:
                json.dump({
                    'text': result['text'],
                    'segments': result['segments'],
                    'language': result.get('language'),
                    'audio_path': audio_path,
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)
                
        elif format == 'txt':
            transcript_path = f"{audio_path}.transcript.txt"
            
            # Save as plain text
            with open(transcript_path, 'w') as f:
                f.write(result['text'])
                
        else:
            logger.warning(f"Unknown transcript format: {format}, defaulting to JSON")
            transcript_path = f"{audio_path}.transcript.json"
            
            with open(transcript_path, 'w') as f:
                json.dump({
                    'text': result['text'],
                    'segments': result['segments']
                }, f, indent=2)
        
        return transcript_path
    
    def scan_for_untranscribed_files(self) -> List[str]:
        """
        Scan for audio files without transcriptions.
        
        Returns:
            List of file paths without transcriptions
        """
        audio_storage = AudioStorage(self.base_path)
        all_files = audio_storage.list_audio_files()
        untranscribed = []
        
        for file_info in all_files:
            audio_path = file_info['path']
            
            # Check if a transcript exists
            json_transcript = f"{audio_path}.transcript.json"
            txt_transcript = f"{audio_path}.transcript.txt"
            
            if not os.path.exists(json_transcript) and not os.path.exists(txt_transcript):
                untranscribed.append(audio_path)
        
        logger.info(f"Found {len(untranscribed)} untranscribed files")
        return untranscribed
    
    def transcribe_all_pending(self) -> None:
        """Queue all untranscribed files for transcription."""
        untranscribed = self.scan_for_untranscribed_files()
        
        with self.queue_lock:
            for audio_path in untranscribed:
                if audio_path not in self.queue:
                    self.queue.append(audio_path)
                    
        logger.info(f"Queued {len(untranscribed)} files for transcription")