
# 🧾 Prompt Plan (prompt_plan.md)

### 📌 1. Mic Monitoring with Voice Detection
```
Build a Python module that listens to microphone input and detects voice using WebRTC VAD.
- It should run continuously.
- When voice is detected, log the timestamp.
- Support basic testing with live mic or pre-recorded audio files.
```

### 📌 2. Pre/Post Buffering Around Speech
```
Given a stream of microphone input and a VAD trigger, implement pre- and post-buffering.
- Maintain a rolling buffer (default 2 mins).
- Once speech is detected, save the 2 minutes before and 2 minutes after as a single segment.
- Return raw PCM data.
```

### 📌 3. Audio Normalization & Encoding
```
Write a function that:
- Takes raw PCM audio,
- Normalizes it for consistent volume,
- Encodes it to Opus (.opus) format using ffmpeg or pydub.
Return the final audio file.
```

### 📌 4. File Naming and Storage
```
Save the Opus audio file to a local folder path structured as /audiolog/YYYY/MM/DD.
Name the file using: [device_id]_[timestamp].opus
Also generate a metadata JSON file with start_time, end_time, device_id.
```

### 📌 5. System Tray Control
```
Add a system tray icon (Windows/macOS) with two menu options:
- "Pause Recording"
- "Resume Recording"
Change the icon to reflect current status.
```

### 📌 6. Google Drive Sync
```
Use Google Drive API to upload all unsynced .opus files in the audiolog folder.
Authenticate with OAuth2.
Avoid re-uploading already synced files.
Retry failed uploads with backoff.
```

### 📌 7. Transcription Pipeline
```
Create a script that:
- Watches the synced folder for new .opus files,
- Runs Whisper (local, offline) to generate text transcripts,
- Saves transcript as .txt or .json next to the audio file.
Make the process optional and configurable.
```

### 📌 8. Audio Conversion Tool
```
Write a command-line tool to convert WAV/MP3/FLAC files to Opus.
Use ffmpeg bindings or CLI.
Preserve original filename with .opus extension.
Save in a user-specified folder.
```

### 📌 9. Config File Loader
```
Create a config loader for user preferences (YAML/JSON):
- Buffers (pre/post)
- Transcription (on/off, trigger mode)
- Google Drive sync (on/off)
- Device name
Make settings hot-reloadable.
```
