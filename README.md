# AudioLog

A system for continuous audio recording with voice detection, buffering, and transcription capabilities.

## Features

- <¤ Continuous microphone monitoring with WebRTC VAD
- = Pre/post buffer to capture speech context (default 2min before/after)
- =
 Audio normalization and Opus encoding
- =Â Organized file storage by date (YYYY/MM/DD)
- =¥ System tray controls for pause/resume
-  Google Drive sync for audio files
- =$ Automatic transcription using local Whisper
- = Audio conversion tool for existing files
- ™ Hot-reloadable configuration

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/audiolog.git
cd audiolog

# Install dependencies
pip install -e .
```

## Usage

### Start AudioLog

```bash
audiolog
```

### Convert existing audio files

```bash
audiolog-convert input.mp3 --output-dir /path/to/output
```

## Configuration

Edit the config file at `~/.config/audiolog/config.yaml`:

```yaml
buffer:
  pre_buffer_seconds: 120  # 2 minutes
  post_buffer_seconds: 120 # 2 minutes

device:
  id: "default_device"
  name: "My Computer"

audio:
  sample_rate: 16000
  channels: 1
  format: "int16"

storage:
  base_path: "~/audiolog"

transcription:
  enabled: true
  model: "tiny"  # tiny, base, small, medium, large

google_drive:
  enabled: true
  credentials_path: "~/.config/audiolog/credentials.json"
  token_path: "~/.config/audiolog/token.json"
```

## Requirements

- Python 3.8+
- FFmpeg
- PyAudio dependencies (PortAudio)
- Whisper dependencies

## License

MIT