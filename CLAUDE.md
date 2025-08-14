# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Installation and Setup
```bash
# Install the package in development mode
pip install -e .

# Install dependencies
pip install -r requirements.txt
```

### Running Tests
```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_buffer.py

# Run with verbose output
python -m pytest -v tests/
```

### Running the Application
```bash
# Start AudioLog with microphone
audiolog

# Start with custom config
audiolog --config configs/demo_config.yaml

# Process a file for testing
audiolog --file test_audio/sample.wav

# Run without UI (headless mode)
audiolog --no-ui

# Enable verbose logging
audiolog --verbose

# Convert existing audio files
audiolog-convert input.mp3 --output-dir /path/to/output
```

### Demo and Simulation Scripts
```bash
# Run basic demo
python demo.py

# Run full demo with all features
python full_demo.py

# Run headless version
python headless_audiolog.py

# Simulate audio recording for testing
python simulate_recording.py
```

## Project Architecture

AudioLog is a continuous audio recording system with voice activity detection, organized around these core components:

### Core Audio Pipeline
- **VoiceDetector** (`audiolog/core/vad.py`): WebRTC-based voice activity detection
- **AudioBuffer** (`audiolog/core/buffer.py`): Rolling pre/post-speech buffer management
- **AudioProcessor** (`audiolog/core/buffer.py`): PCM audio frame processing

### Audio Processing Chain
1. **MicrophoneAudioStream/FileAudioStream** (`audiolog/core/vad.py`): Audio input streams
2. **VoiceDetector**: Detects speech in 30ms frames
3. **AudioBuffer**: Maintains 2-minute pre/post-speech buffers
4. **AudioNormalizer** (`audiolog/audio/processor.py`): Audio normalization
5. **AudioEncoder** (`audiolog/audio/processor.py`): Opus encoding
6. **AudioStorage** (`audiolog/audio/storage.py`): File organization (YYYY/MM/DD structure)

### Key Services
- **TranscriptionManager** (`audiolog/transcription/whisper.py`): Automatic Whisper-based transcription with file watching
- **GoogleDriveSync** (`audiolog/sync/drive.py`): Optional cloud synchronization
- **System Tray UI** (`audiolog/ui/tray.py`): Pause/resume controls

### Configuration System
- **ConfigManager** (`audiolog/utils/config.py`): Hot-reloadable YAML configuration
- Default config location: `~/.config/audiolog/config.yaml`
- Override with custom configs in `configs/` directory

### File Organization
Audio files are stored with the naming pattern:
```
~/audiolog/YYYY/MM/DD/[device_id]_[timestamp].opus
~/audiolog/YYYY/MM/DD/[device_id]_[timestamp].opus.json (metadata)
~/audiolog/YYYY/MM/DD/[device_id]_[timestamp].opus.transcript.json (transcription)
```

### Application State Management
The main application uses a singleton `AudioLogState` class (`audiolog/cli.py`) that coordinates all components and handles lifecycle management.

### Testing Structure
Tests are organized by component:
- `test_buffer.py`: Audio buffer functionality
- `test_vad.py`: Voice activity detection
- `test_storage.py`: File storage operations
- `test_config.py`: Configuration management
- `test_audio_processor.py`: Audio processing pipeline

### Configuration Presets
- `configs/default_config.yaml`: Standard configuration
- `configs/demo_config.yaml`: Demo mode settings
- `configs/headless_config.yaml`: Server/headless deployment

The system is designed for continuous operation with graceful handling of pause/resume, file watching for transcription, and modular component architecture.