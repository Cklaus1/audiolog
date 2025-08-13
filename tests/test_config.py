#!/usr/bin/env python3
"""
Test for the Config module.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add parent directory to the path to import audiolog
sys.path.insert(0, str(Path(__file__).parent.parent))

from audiolog.utils.config import ConfigManager


def test_config_basics():
    """Test basic config functionality."""
    print("Testing basic config functionality...")
    
    # Create a temporary config file
    with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as tmp:
        tmp_path = tmp.name
        tmp.write(b"""
buffer:
  pre_buffer_seconds: 60
  post_buffer_seconds: 30

device:
  id: "test_device"
  name: "Test Computer"

audio:
  sample_rate: 16000
  channels: 1
""")
    
    try:
        # Create config manager with the temp file
        config = ConfigManager(tmp_path)
        
        # Test getting values
        pre_buffer = config.get('buffer.pre_buffer_seconds')
        post_buffer = config.get('buffer.post_buffer_seconds')
        device_id = config.get('device.id')
        sample_rate = config.get('audio.sample_rate')
        
        print(f"Config values: pre_buffer={pre_buffer}, post_buffer={post_buffer}")
        print(f"Config values: device_id={device_id}, sample_rate={sample_rate}")
        
        # Test getting default for missing value
        missing_value = config.get('missing.key', 'default_value')
        print(f"Missing value with default: {missing_value}")
        
        # Test setting a value
        config.set('device.name', 'Updated Name', save_file=True)
        updated_name = config.get('device.name')
        print(f"Updated name: {updated_name}")
        
        # Success if we got here
        print("Config test successful!")
        return True
    
    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_default_config():
    """Test the default configuration."""
    print("Testing default configuration...")
    
    # Create config manager with no file path (uses default)
    config = ConfigManager()
    
    # Check some default values
    print(f"Default pre_buffer_seconds: {config.get('buffer.pre_buffer_seconds')}")
    print(f"Default post_buffer_seconds: {config.get('buffer.post_buffer_seconds')}")
    print(f"Default device.id: {config.get('device.id')}")
    print(f"Default storage.base_path: {config.get('storage.base_path')}")
    
    # Check that configuration was saved
    config_path = config.config_path
    print(f"Default config path: {config_path}")
    
    if os.path.exists(config_path):
        print(f"Default config file was created successfully")
    else:
        print(f"WARNING: Default config file was not created")
    
    return True


def main():
    """Main entry point."""
    print("=== AudioLog Configuration Tests ===\n")
    
    # Run tests
    test_config_basics()
    print("\n")
    test_default_config()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())