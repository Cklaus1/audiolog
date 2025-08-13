import os
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "buffer": {
        "pre_buffer_seconds": 120,  # 2 minutes
        "post_buffer_seconds": 120,  # 2 minutes
    },
    "device": {
        "id": "default_device",
        "name": os.uname().nodename,
    },
    "audio": {
        "sample_rate": 16000,
        "channels": 1,
        "format": "int16",
        "chunk_size": 480,  # 30ms at 16kHz
    },
    "storage": {
        "base_path": str(Path.home() / "audiolog"),
    },
    "transcription": {
        "enabled": True,
        "model": "tiny",  # tiny, base, small, medium, large
        "format": "json",  # json or txt
    },
    "google_drive": {
        "enabled": False,
        "credentials_path": str(Path.home() / ".config" / "audiolog" / "credentials.json"),
        "token_path": str(Path.home() / ".config" / "audiolog" / "token.json"),
    },
    "ui": {
        "enabled": True,
        "show_notifications": True,
    }
}


class ConfigChangeHandler(FileSystemEventHandler):
    def __init__(self, config_manager):
        self.config_manager = config_manager
        
    def on_modified(self, event):
        if not event.is_directory and event.src_path == self.config_manager.config_path:
            logger.info(f"Config file changed: {event.src_path}")
            self.config_manager.reload()


class ConfigManager:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, config_path: Optional[str] = None):
        if not hasattr(self, 'initialized'):
            if config_path is None:
                config_dir = Path.home() / ".config" / "audiolog"
                config_dir.mkdir(parents=True, exist_ok=True)
                self.config_path = str(config_dir / "config.yaml")
            else:
                self.config_path = config_path
                
            self.config = DEFAULT_CONFIG.copy()
            self.observer = None
            self.initialized = True
            self.load()
            self.start_watching()
    
    def load(self) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                        user_config = yaml.safe_load(f)
                    elif self.config_path.endswith('.json'):
                        user_config = json.load(f)
                    else:
                        raise ValueError(f"Unsupported config file format: {self.config_path}")
                    
                # Update config with user values, keeping defaults for missing values
                self._update_recursive(self.config, user_config)
                logger.info(f"Loaded configuration from {self.config_path}")
            else:
                # Save default config if no file exists
                self.save()
                logger.info(f"Created default configuration at {self.config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
        
        return self.config
    
    def _update_recursive(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """Recursively update target dict with values from source dict."""
        if source is None:
            return
            
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                # Recursively update nested dicts
                self._update_recursive(target[key], value)
            else:
                # Update or add the value
                target[key] = value
    
    def save(self) -> None:
        """Save current configuration to file."""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            with open(self.config_path, 'w') as f:
                if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                    yaml.dump(self.config, f, default_flow_style=False)
                elif self.config_path.endswith('.json'):
                    json.dump(self.config, f, indent=2)
                else:
                    raise ValueError(f"Unsupported config file format: {self.config_path}")
                    
            logger.info(f"Saved configuration to {self.config_path}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def reload(self) -> Dict[str, Any]:
        """Reload configuration from file."""
        logger.info("Reloading configuration")
        return self.load()
    
    def start_watching(self) -> None:
        """Start watching config file for changes."""
        if self.observer is None:
            self.observer = Observer()
            self.observer.schedule(
                ConfigChangeHandler(self),
                os.path.dirname(self.config_path),
                recursive=False
            )
            self.observer.start()
            logger.info(f"Started watching config file: {self.config_path}")
    
    def stop_watching(self) -> None:
        """Stop watching config file for changes."""
        if self.observer is not None:
            self.observer.stop()
            self.observer.join()
            self.observer = None
            logger.info("Stopped watching config file")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a config value by key with dot notation (e.g. 'buffer.pre_buffer_seconds')."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
    
    def set(self, key: str, value: Any, save_file: bool = True) -> None:
        """Set a config value by key with dot notation."""
        keys = key.split('.')
        target = self.config
        
        # Navigate to the nested dict
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]
        
        # Set the value
        target[keys[-1]] = value
        
        if save_file:
            self.save()


# Create a global instance
config = ConfigManager()