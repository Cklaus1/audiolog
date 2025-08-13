from setuptools import setup, find_packages

setup(
    name="audiolog",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pyaudio>=0.2.11",
        "webrtcvad>=2.0.10",
        "numpy>=1.19.0",
        "pydub>=0.25.1",
        "ffmpeg-python>=0.2.0",
        "watchdog>=2.1.0",
        "librosa>=0.8.0",
        "google-auth>=2.0.0",
        "google-auth-oauthlib>=0.4.6",
        "google-api-python-client>=2.0.0",
        "openai-whisper>=20230124",
        "pystray>=0.19.0",
        "Pillow>=8.0.0",
        "pyyaml>=6.0",
    ],
    entry_points={
        "console_scripts": [
            "audiolog=audiolog.cli:main",
            "audiolog-convert=audiolog.audio.converter:main",
        ],
    },
    python_requires=">=3.8",
    author="AudioLog Team",
    author_email="info@audiolog.example.com",
    description="A system for continuous audio recording with voice detection",
    keywords="audio, recording, voice, transcription",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: End Users/Desktop",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)