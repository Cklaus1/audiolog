#!/usr/bin/env python3
"""
Test for the AudioStorage module.
"""

import os
import sys
import time
import json
import tempfile
from pathlib import Path
from datetime import datetime

# Add parent directory to the path to import audiolog
sys.path.insert(0, str(Path(__file__).parent.parent))

from audiolog.audio.storage import AudioStorage


def create_dummy_audio_file():
    """Create a dummy opus file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.opus', delete=False) as tmp:
        # Just create an empty file
        tmp.write(b'dummy audio data')
        return tmp.name


def test_storage_paths():
    """Test storage path generation."""
    print("Testing storage path generation...")
    
    # Create a temp directory for storage
    with tempfile.TemporaryDirectory() as temp_dir:
        storage = AudioStorage(temp_dir)
        
        # Test with current time
        now = datetime.now()
        path = storage.get_storage_path(now)
        
        # Verify path structure
        expected_path = Path(temp_dir) / f"{now.year}" / f"{now.month:02d}" / f"{now.day:02d}"
        if path == expected_path:
            print(f"Path structure correct: {path}")
        else:
            print(f"ERROR: Path structure incorrect")
            print(f"Expected: {expected_path}")
            print(f"Got: {path}")
            
        # Check that directory was created
        if path.exists() and path.is_dir():
            print("Directory was created successfully")
        else:
            print("ERROR: Directory was not created")
            
    return True


def test_filename_generation():
    """Test filename generation."""
    print("Testing filename generation...")
    
    storage = AudioStorage()
    
    # Test with specific values
    device_id = "test_device"
    timestamp = datetime(2023, 5, 18, 14, 30, 45)
    
    filename = storage.generate_filename(device_id, timestamp)
    expected = "test_device_20230518T143045.opus"
    
    if filename == expected:
        print(f"Filename generated correctly: {filename}")
    else:
        print(f"ERROR: Filename incorrect")
        print(f"Expected: {expected}")
        print(f"Got: {filename}")
    
    return True


def test_save_file():
    """Test saving an audio file."""
    print("Testing file saving...")
    
    # Create a temp directory for storage
    with tempfile.TemporaryDirectory() as temp_dir:
        storage = AudioStorage(temp_dir)
        
        # Create a dummy audio file
        dummy_file = create_dummy_audio_file()
        
        try:
            # Create metadata
            device_id = "test_device"
            timestamp = datetime.now()
            start_time = timestamp.timestamp() - 10  # 10 seconds ago
            end_time = timestamp.timestamp()
            
            metadata = {
                "device_id": device_id,
                "start_time": start_time,
                "end_time": end_time,
                "duration": end_time - start_time
            }
            
            # Save the file
            audio_path, metadata_path = storage.save_audio_file(
                dummy_file, 
                metadata,
                device_id=device_id,
                timestamp=timestamp
            )
            
            print(f"Saved audio file: {audio_path}")
            print(f"Saved metadata: {metadata_path}")
            
            # Verify files exist
            if os.path.exists(audio_path) and os.path.exists(metadata_path):
                print("Files were saved successfully")
                
                # Verify metadata content
                with open(metadata_path, 'r') as f:
                    saved_metadata = json.load(f)
                    
                print(f"Metadata content: {saved_metadata}")
                
                if saved_metadata.get('device_id') == device_id:
                    print("Metadata content is correct")
                else:
                    print("ERROR: Metadata content is incorrect")
            else:
                print("ERROR: Files were not saved correctly")
            
        finally:
            # Clean up the dummy file if it still exists
            if os.path.exists(dummy_file):
                os.unlink(dummy_file)
    
    return True


def test_list_files():
    """Test listing audio files."""
    print("Testing file listing...")
    
    # Create a temp directory for storage
    with tempfile.TemporaryDirectory() as temp_dir:
        storage = AudioStorage(temp_dir)
        
        # Create and save a few files
        for i in range(3):
            dummy_file = create_dummy_audio_file()
            
            try:
                device_id = f"device_{i}"
                timestamp = datetime.now()
                
                metadata = {
                    "device_id": device_id,
                    "start_time": timestamp.timestamp() - 10,
                    "end_time": timestamp.timestamp(),
                    "test_value": f"test_{i}"
                }
                
                storage.save_audio_file(
                    dummy_file, 
                    metadata,
                    device_id=device_id
                )
                
            finally:
                if os.path.exists(dummy_file):
                    os.unlink(dummy_file)
                    
        # Wait a moment to ensure files are saved
        time.sleep(0.1)
            
        # List files without metadata
        files = storage.list_audio_files(include_metadata=False)
        print(f"Found {len(files)} files")
        
        if len(files) == 3:
            print("Correct number of files found")
        else:
            print(f"ERROR: Expected 3 files, got {len(files)}")
            
        # List files with metadata
        files_with_metadata = storage.list_audio_files(include_metadata=True)
        
        for i, file_info in enumerate(files_with_metadata):
            print(f"File {i+1}: {file_info['filename']}")
            if 'metadata' in file_info:
                print(f"  Metadata: {file_info['metadata']}")
                
                # Verify test values
                if 'test_value' in file_info['metadata']:
                    print(f"  Test value: {file_info['metadata']['test_value']}")
                else:
                    print("  ERROR: Missing test value in metadata")
            else:
                print("  ERROR: Missing metadata")
    
    return True


def main():
    """Main entry point."""
    print("=== AudioLog Storage Tests ===\n")
    
    # Run tests
    tests = [
        test_storage_paths,
        test_filename_generation,
        test_save_file,
        test_list_files
    ]
    
    for i, test in enumerate(tests):
        print(f"\n--- Test {i+1}: {test.__name__} ---")
        test()
    
    print("\nAll tests completed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())