import os
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union

from audiolog.utils.config import config

logger = logging.getLogger(__name__)

class AudioStorage:
    """
    Handle audio file storage and organization.
    Implements the file naming and folder structure according to specifications.
    """
    
    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize the audio storage.
        
        Args:
            base_path: Base directory to store audio files (defaults to config value)
        """
        if base_path is None:
            base_path = config.get('storage.base_path')
            
        # Expand user path if needed
        if '~' in base_path:
            base_path = os.path.expanduser(base_path)
            
        self.base_path = Path(base_path)
        logger.debug(f"Initialized AudioStorage with base path: {self.base_path}")
    
    def get_storage_path(self, timestamp: Optional[datetime] = None) -> Path:
        """
        Get the storage path for a specific timestamp.
        Creates directories as needed.
        
        Args:
            timestamp: Datetime object (default: current time)
            
        Returns:
            Path object for the storage directory
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        # Create path with YYYY/MM/DD structure
        storage_path = self.base_path / f"{timestamp.year}" / f"{timestamp.month:02d}" / f"{timestamp.day:02d}"
        storage_path.mkdir(parents=True, exist_ok=True)
        
        return storage_path
    
    def generate_filename(self, device_id: Optional[str] = None, timestamp: Optional[datetime] = None) -> str:
        """
        Generate a filename for an audio file based on device ID and timestamp.
        
        Args:
            device_id: Device identifier (default: from config)
            timestamp: Datetime object (default: current time)
            
        Returns:
            Filename in the format [device_id]_[timestamp].opus
        """
        if device_id is None:
            device_id = config.get('device.id', 'default_device')
            
        if timestamp is None:
            timestamp = datetime.now()
            
        # Format timestamp as ISO 8601 with some modifications for filename safety
        timestamp_str = timestamp.strftime("%Y%m%dT%H%M%S")
        
        return f"{device_id}_{timestamp_str}.opus"
    
    def save_audio_file(self, 
                        audio_path: str, 
                        metadata: Dict[str, Any],
                        device_id: Optional[str] = None,
                        timestamp: Optional[datetime] = None) -> Tuple[str, str]:
        """
        Move an audio file to the storage location and save its metadata.
        
        Args:
            audio_path: Path to the audio file
            metadata: Metadata to save with the file
            device_id: Device identifier (default: from config)
            timestamp: Datetime object (default: from metadata or current time)
            
        Returns:
            Tuple of (audio_file_path, metadata_file_path)
        """
        # Get timestamp from metadata or use current time
        if timestamp is None:
            if 'start_time' in metadata and isinstance(metadata['start_time'], (int, float)):
                timestamp = datetime.fromtimestamp(metadata['start_time'])
            else:
                timestamp = datetime.now()
                
        # Get storage path and generate filename
        storage_path = self.get_storage_path(timestamp)
        filename = self.generate_filename(device_id, timestamp)
        
        # Destination paths
        audio_file_path = storage_path / filename
        metadata_file_path = storage_path / f"{filename}.json"
        
        # Copy audio file if it's not already in the right location
        if os.path.abspath(audio_path) != os.path.abspath(audio_file_path):
            logger.debug(f"Moving audio file from {audio_path} to {audio_file_path}")
            os.replace(audio_path, audio_file_path)
        
        # Save metadata
        with open(metadata_file_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Saved audio file: {audio_file_path}")
        logger.debug(f"Saved metadata: {metadata_file_path}")
        
        return str(audio_file_path), str(metadata_file_path)
    
    def list_audio_files(self, include_metadata: bool = False) -> List[Dict[str, Any]]:
        """
        List all audio files in the storage.
        
        Args:
            include_metadata: Whether to include metadata in the results
            
        Returns:
            List of dictionaries with file info
        """
        if not self.base_path.exists():
            return []
            
        results = []
        
        # Walk through the directory structure
        for year_dir in sorted(self.base_path.glob("*"), reverse=True):
            if not year_dir.is_dir() or not year_dir.name.isdigit():
                continue
                
            for month_dir in sorted(year_dir.glob("*"), reverse=True):
                if not month_dir.is_dir() or not month_dir.name.isdigit():
                    continue
                    
                for day_dir in sorted(month_dir.glob("*"), reverse=True):
                    if not day_dir.is_dir() or not day_dir.name.isdigit():
                        continue
                        
                    for audio_file in sorted(day_dir.glob("*.opus"), reverse=True):
                        file_info = {
                            "path": str(audio_file),
                            "filename": audio_file.name,
                            "size": audio_file.stat().st_size,
                            "created": audio_file.stat().st_ctime,
                            "modified": audio_file.stat().st_mtime,
                        }
                        
                        # Parse device ID and timestamp from filename
                        filename_parts = audio_file.stem.split('_', 1)
                        if len(filename_parts) == 2:
                            file_info["device_id"] = filename_parts[0]
                            file_info["timestamp"] = filename_parts[1]
                        
                        # Include metadata if requested
                        if include_metadata:
                            metadata_path = audio_file.with_suffix('.opus.json')
                            if metadata_path.exists():
                                try:
                                    with open(metadata_path, 'r') as f:
                                        file_info["metadata"] = json.load(f)
                                except Exception as e:
                                    logger.warning(f"Error reading metadata file {metadata_path}: {e}")
                        
                        results.append(file_info)
        
        return results
    
    def find_audio_file(self, device_id: str, timestamp: Union[datetime, str]) -> Optional[Dict[str, Any]]:
        """
        Find a specific audio file by device ID and timestamp.
        
        Args:
            device_id: Device identifier
            timestamp: Datetime object or timestamp string
            
        Returns:
            Dictionary with file info if found, None otherwise
        """
        # Convert timestamp to datetime if it's a string
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp.replace('T', ' '))
            except ValueError:
                # Try different format if ISO format fails
                try:
                    timestamp = datetime.strptime(timestamp, "%Y%m%dT%H%M%S")
                except ValueError:
                    logger.error(f"Invalid timestamp format: {timestamp}")
                    return None
        
        # Get the expected storage path
        storage_path = self.get_storage_path(timestamp)
        
        # Generate the expected filename
        filename = self.generate_filename(device_id, timestamp)
        
        # Look for the file
        audio_path = storage_path / filename
        if audio_path.exists():
            file_info = {
                "path": str(audio_path),
                "filename": audio_path.name,
                "size": audio_path.stat().st_size,
                "created": audio_path.stat().st_ctime,
                "modified": audio_path.stat().st_mtime,
                "device_id": device_id,
                "timestamp": timestamp.strftime("%Y%m%dT%H%M%S"),
            }
            
            # Include metadata if available
            metadata_path = audio_path.with_suffix('.opus.json')
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        file_info["metadata"] = json.load(f)
                except Exception as e:
                    logger.warning(f"Error reading metadata file {metadata_path}: {e}")
            
            return file_info
        
        return None