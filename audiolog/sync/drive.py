import os
import json
import time
import logging
import hashlib
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timedelta

import google.auth.exceptions
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

from audiolog.utils.config import config
from audiolog.audio.storage import AudioStorage

logger = logging.getLogger(__name__)

# Define the scopes required for Drive API access
SCOPES = ['https://www.googleapis.com/auth/drive.file']


class GoogleDriveSync:
    """
    Syncs audio files to Google Drive.
    Handles authentication, file upload, and tracking of synced files.
    """
    
    def __init__(self, 
                 credentials_path: Optional[str] = None,
                 token_path: Optional[str] = None,
                 base_folder_name: str = "AudioLog",
                 sync_interval_seconds: int = 300):  # 5 minutes
        """
        Initialize the Google Drive sync.
        
        Args:
            credentials_path: Path to the credentials.json file
            token_path: Path to the token.json file
            base_folder_name: Name for the base folder in Google Drive
            sync_interval_seconds: Interval between sync attempts
        """
        # Get configuration
        if credentials_path is None:
            credentials_path = config.get('google_drive.credentials_path')
            
        if token_path is None:
            token_path = config.get('google_drive.token_path')
        
        # Expand user paths if needed
        if '~' in credentials_path:
            credentials_path = os.path.expanduser(credentials_path)
            
        if '~' in token_path:
            token_path = os.path.expanduser(token_path)
            
        self.credentials_path = credentials_path
        self.token_path = token_path
        self.base_folder_name = base_folder_name
        self.sync_interval_seconds = sync_interval_seconds
        
        # Initialize state
        self.credentials = None
        self.service = None
        self.synced_files: Set[str] = set()
        self.sync_active = False
        self.sync_thread = None
        self.audio_storage = AudioStorage()
        self.syncing_lock = threading.RLock()
        self.last_sync_time = 0
        
        # Load synced files registry if exists
        self._load_sync_registry()
        
        logger.debug(f"Initialized GoogleDriveSync (credentials={credentials_path}, token={token_path})")
    
    def authenticate(self) -> bool:
        """
        Authenticate with Google Drive.
        
        Returns:
            True if authentication successful, False otherwise
        """
        try:
            # Check if token exists
            credentials = None
            if os.path.exists(self.token_path):
                try:
                    credentials = Credentials.from_authorized_user_info(
                        json.loads(Path(self.token_path).read_text()), 
                        SCOPES
                    )
                except Exception as e:
                    logger.warning(f"Error loading token: {e}")
            
            # If credentials are invalid or expired
            if not credentials or not credentials.valid:
                if credentials and credentials.expired and credentials.refresh_token:
                    # Refresh token if possible
                    credentials.refresh(Request())
                else:
                    # Start OAuth flow if credentials invalid
                    if not os.path.exists(self.credentials_path):
                        logger.error(f"Credentials file not found: {self.credentials_path}")
                        return False
                        
                    # Create new credentials
                    flow = InstalledAppFlow.from_client_secrets_file(
                        self.credentials_path, 
                        SCOPES
                    )
                    credentials = flow.run_local_server(port=0)
                
                # Save the new token
                os.makedirs(os.path.dirname(self.token_path), exist_ok=True)
                with open(self.token_path, 'w') as token_file:
                    token_file.write(credentials.to_json())
                    logger.info(f"Saved authentication token to {self.token_path}")
            
            # Create Drive API service
            self.credentials = credentials
            self.service = build('drive', 'v3', credentials=credentials)
            logger.info("Successfully authenticated with Google Drive")
            return True
            
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return False
    
    def _get_folder_id(self, folder_name: str, parent_id: Optional[str] = None) -> Optional[str]:
        """
        Get a folder ID, creating the folder if it doesn't exist.
        
        Args:
            folder_name: Name of the folder
            parent_id: ID of the parent folder
            
        Returns:
            Folder ID or None if failed
        """
        try:
            # Build query
            query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'"
            if parent_id:
                query += f" and '{parent_id}' in parents"
            
            # Search for the folder
            response = self.service.files().list(
                q=query,
                spaces='drive',
                fields='files(id, name)'
            ).execute()
            
            folders = response.get('files', [])
            
            # Return the folder ID if found
            if folders:
                logger.debug(f"Found folder: {folder_name} (ID: {folders[0]['id']})")
                return folders[0]['id']
            
            # Create the folder if not found
            folder_metadata = {
                'name': folder_name,
                'mimeType': 'application/vnd.google-apps.folder'
            }
            
            if parent_id:
                folder_metadata['parents'] = [parent_id]
                
            folder = self.service.files().create(
                body=folder_metadata,
                fields='id'
            ).execute()
            
            logger.info(f"Created folder: {folder_name} (ID: {folder['id']})")
            return folder['id']
            
        except HttpError as e:
            logger.error(f"Error getting/creating folder {folder_name}: {e}")
            return None
    
    def _ensure_folder_structure(self, timestamp: datetime) -> Optional[str]:
        """
        Ensure the folder structure exists for a given timestamp.
        Creates year/month/day folders if needed.
        
        Args:
            timestamp: Datetime object
            
        Returns:
            Folder ID for the day folder or None if failed
        """
        try:
            # Get or create base folder
            base_folder_id = self._get_folder_id(self.base_folder_name)
            if not base_folder_id:
                return None
                
            # Get or create year folder
            year_folder_name = str(timestamp.year)
            year_folder_id = self._get_folder_id(year_folder_name, base_folder_id)
            if not year_folder_id:
                return None
                
            # Get or create month folder
            month_folder_name = f"{timestamp.month:02d}"
            month_folder_id = self._get_folder_id(month_folder_name, year_folder_id)
            if not month_folder_id:
                return None
                
            # Get or create day folder
            day_folder_name = f"{timestamp.day:02d}"
            day_folder_id = self._get_folder_id(day_folder_name, month_folder_id)
            
            return day_folder_id
            
        except Exception as e:
            logger.error(f"Error ensuring folder structure: {e}")
            return None
    
    def upload_file(self, file_path: str, timestamp: Optional[datetime] = None) -> bool:
        """
        Upload a file to Google Drive.
        
        Args:
            file_path: Path to the file to upload
            timestamp: Datetime object (default: extracted from filename)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return False
                
            # Extract timestamp from filename if not provided
            if timestamp is None:
                try:
                    filename = os.path.basename(file_path)
                    device_id, timestamp_str = filename.split('_', 1)
                    timestamp_str = timestamp_str.split('.')[0]  # Remove extension
                    timestamp = datetime.strptime(timestamp_str, "%Y%m%dT%H%M%S")
                except Exception as e:
                    logger.warning(f"Could not extract timestamp from filename: {e}")
                    timestamp = datetime.now()
            
            # Ensure folder structure exists
            day_folder_id = self._ensure_folder_structure(timestamp)
            if not day_folder_id:
                logger.error("Failed to create folder structure")
                return False
            
            # Upload the file
            file_name = os.path.basename(file_path)
            file_metadata = {
                'name': file_name,
                'parents': [day_folder_id]
            }
            
            mime_type = 'audio/opus'  # Opus file mime type
            media = MediaFileUpload(file_path, mimetype=mime_type, resumable=True)
            
            logger.debug(f"Uploading file: {file_path}")
            file = self.service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            ).execute()
            
            # Also upload metadata file if exists
            metadata_path = f"{file_path}.json"
            if os.path.exists(metadata_path):
                metadata_file_name = os.path.basename(metadata_path)
                metadata_file_metadata = {
                    'name': metadata_file_name,
                    'parents': [day_folder_id]
                }
                
                metadata_media = MediaFileUpload(
                    metadata_path, 
                    mimetype='application/json', 
                    resumable=True
                )
                
                self.service.files().create(
                    body=metadata_file_metadata,
                    media_body=metadata_media,
                    fields='id'
                ).execute()
            
            # Add to synced files
            self._add_synced_file(file_path)
            
            logger.info(f"Successfully uploaded {file_path} to Google Drive")
            return True
            
        except HttpError as e:
            logger.error(f"Error uploading file {file_path}: {e}")
            
            # Retry for certain errors
            if e.resp.status in [403, 429, 500, 502, 503, 504]:
                logger.info(f"Upload failed with status {e.resp.status}, will retry later")
                return False
                
            # For other errors, mark as synced to avoid retrying
            self._add_synced_file(file_path)
            return False
            
        except Exception as e:
            logger.error(f"Unexpected error uploading file {file_path}: {e}")
            return False
    
    def _add_synced_file(self, file_path: str) -> None:
        """
        Add a file to the synced files registry.
        
        Args:
            file_path: Path to the synced file
        """
        file_path = os.path.abspath(file_path)
        
        with self.syncing_lock:
            self.synced_files.add(file_path)
            self._save_sync_registry()
    
    def _load_sync_registry(self) -> None:
        """Load the synced files registry from disk."""
        registry_path = self._get_registry_path()
        
        try:
            if os.path.exists(registry_path):
                with open(registry_path, 'r') as f:
                    data = json.load(f)
                    self.synced_files = set(data.get('synced_files', []))
                    logger.debug(f"Loaded {len(self.synced_files)} synced files from registry")
        except Exception as e:
            logger.error(f"Error loading sync registry: {e}")
            self.synced_files = set()
    
    def _save_sync_registry(self) -> None:
        """Save the synced files registry to disk."""
        registry_path = self._get_registry_path()
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(registry_path), exist_ok=True)
            
            # Save registry
            with open(registry_path, 'w') as f:
                json.dump({
                    'synced_files': list(self.synced_files),
                    'last_update': datetime.now().isoformat()
                }, f, indent=2)
                
            logger.debug(f"Saved sync registry with {len(self.synced_files)} files")
        except Exception as e:
            logger.error(f"Error saving sync registry: {e}")
    
    def _get_registry_path(self) -> str:
        """
        Get the path to the sync registry file.
        
        Returns:
            Path to the registry file
        """
        config_dir = os.path.dirname(self.token_path)
        return os.path.join(config_dir, 'synced_files.json')
    
    def sync_files(self) -> int:
        """
        Sync unsynced audio files to Google Drive.
        
        Returns:
            Number of files synced
        """
        with self.syncing_lock:
            # Track synced files
            files_synced = 0
            
            # First make sure we're authenticated
            if not self.service and not self.authenticate():
                logger.error("Authentication failed, cannot sync files")
                return 0
            
            try:
                # Get all audio files
                logger.debug("Scanning for unsynced audio files")
                audio_files = self.audio_storage.list_audio_files()
                
                # Filter for unsynced files
                unsynced_files = [
                    file for file in audio_files
                    if os.path.abspath(file['path']) not in self.synced_files
                ]
                
                if not unsynced_files:
                    logger.debug("No new files to sync")
                    return 0
                
                logger.info(f"Found {len(unsynced_files)} files to sync")
                
                # Upload each file
                for file_info in unsynced_files:
                    file_path = file_info['path']
                    
                    # Extract timestamp from filename
                    try:
                        timestamp = datetime.strptime(file_info['timestamp'], "%Y%m%dT%H%M%S")
                    except (KeyError, ValueError):
                        timestamp = None
                    
                    # Try to upload with exponential backoff
                    max_retries = 3
                    retry_delay = 2  # starting delay in seconds
                    
                    for retry in range(max_retries):
                        if self.upload_file(file_path, timestamp):
                            files_synced += 1
                            break
                        else:
                            # Exponential backoff
                            retry_delay *= 2
                            logger.warning(f"Upload failed, retrying in {retry_delay}s (attempt {retry+1}/{max_retries})")
                            time.sleep(retry_delay)
                
                logger.info(f"Synced {files_synced} files to Google Drive")
                return files_synced
                
            except Exception as e:
                logger.error(f"Error during file sync: {e}")
                return files_synced
    
    def start_sync_thread(self) -> None:
        """Start the background sync thread."""
        if not config.get('google_drive.enabled', False):
            logger.info("Google Drive sync is disabled in configuration")
            return
            
        if self.sync_thread and self.sync_thread.is_alive():
            logger.warning("Sync thread is already running")
            return
            
        self.sync_active = True
        self.sync_thread = threading.Thread(
            target=self._sync_thread_worker,
            daemon=True
        )
        self.sync_thread.start()
        logger.info("Started Google Drive sync thread")
    
    def stop_sync_thread(self) -> None:
        """Stop the background sync thread."""
        self.sync_active = False
        if self.sync_thread and self.sync_thread.is_alive():
            self.sync_thread.join(timeout=5.0)
            logger.info("Stopped Google Drive sync thread")
    
    def _sync_thread_worker(self) -> None:
        """Worker function for the sync thread."""
        while self.sync_active:
            try:
                # Check if it's time to sync
                now = time.time()
                time_since_last_sync = now - self.last_sync_time
                
                if time_since_last_sync >= self.sync_interval_seconds:
                    logger.debug("Starting scheduled sync")
                    self.sync_files()
                    self.last_sync_time = time.time()
                
                # Sleep for a bit
                time.sleep(min(10, self.sync_interval_seconds / 10))
            except Exception as e:
                logger.error(f"Error in sync thread: {e}")
                time.sleep(30)  # Sleep longer after an error