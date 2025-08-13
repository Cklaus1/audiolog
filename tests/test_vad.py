#!/usr/bin/env python3
"""
Simple test for the Voice Activity Detection module.
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path

# Add parent directory to the path to import audiolog
sys.path.insert(0, str(Path(__file__).parent.parent))

from audiolog.core.vad import VoiceDetector, MicrophoneAudioStream, FileAudioStream


def setup_logging(verbose=False):
    """Set up basic logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )


def test_mic_vad(duration=10, aggressiveness=3):
    """
    Test VAD with microphone input.
    
    Args:
        duration: Test duration in seconds
        aggressiveness: VAD aggressiveness (0-3)
    """
    print(f"Testing microphone VAD for {duration} seconds (aggressiveness={aggressiveness})...")
    
    # Create VAD
    vad = VoiceDetector(
        aggressiveness=aggressiveness,
        sample_rate=16000,
        frame_duration_ms=30
    )
    
    # Create microphone stream
    with MicrophoneAudioStream(
        sample_rate=16000,
        channels=1,
        chunk_size=480  # 30ms at 16kHz
    ) as mic:
        # Get the stream
        stream = mic.get_stream()
        
        # Process audio frames
        start_time = time.time()
        speech_detected = False
        speech_frames = 0
        total_frames = 0
        
        while time.time() - start_time < duration:
            try:
                # Get frame
                frame = next(stream)
                total_frames += 1
                
                # Detect speech
                is_speech = vad.is_speech(frame)
                
                if is_speech:
                    speech_frames += 1
                    if not speech_detected:
                        speech_detected = True
                        print(f"Speech detected at {time.time() - start_time:.2f}s")
                elif speech_detected:
                    speech_detected = False
                    print(f"Speech ended at {time.time() - start_time:.2f}s")
                    
            except StopIteration:
                break
                
        # Print results
        speech_percentage = (speech_frames / total_frames) * 100 if total_frames > 0 else 0
        print(f"Test complete. Speech detected in {speech_percentage:.1f}% of frames")


def test_file_vad(file_path, aggressiveness=3):
    """
    Test VAD with an audio file.
    
    Args:
        file_path: Path to the audio file
        aggressiveness: VAD aggressiveness (0-3)
    """
    print(f"Testing file VAD with {file_path} (aggressiveness={aggressiveness})...")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return
    
    # Create VAD
    vad = VoiceDetector(
        aggressiveness=aggressiveness,
        sample_rate=16000,
        frame_duration_ms=30
    )
    
    # Create file stream
    with FileAudioStream(
        file_path=file_path,
        chunk_size=480  # 30ms at 16kHz
    ) as file_stream:
        # Get the stream
        stream = file_stream.get_stream()
        
        # Process audio frames
        speech_detected = False
        speech_frames = 0
        total_frames = 0
        speech_segments = []
        current_segment = None
        
        for frame in stream:
            total_frames += 1
            
            # Detect speech
            is_speech = vad.is_speech(frame)
            
            if is_speech:
                speech_frames += 1
                
                if not speech_detected:
                    speech_detected = True
                    current_segment = total_frames
                    print(f"Speech detected at frame {total_frames}")
            elif speech_detected:
                speech_detected = False
                print(f"Speech ended at frame {total_frames} (duration: {total_frames - current_segment} frames)")
                
                if current_segment is not None:
                    speech_segments.append((current_segment, total_frames))
                    current_segment = None
        
        # Print results
        speech_percentage = (speech_frames / total_frames) * 100 if total_frames > 0 else 0
        print(f"Test complete. Processed {total_frames} frames.")
        print(f"Speech detected in {speech_percentage:.1f}% of frames")
        print(f"Speech segments: {len(speech_segments)}")
        
        for i, (start, end) in enumerate(speech_segments):
            duration_sec = (end - start) * (30 / 1000)  # 30ms frames
            print(f"  Segment {i+1}: frames {start}-{end} (duration: {duration_sec:.2f}s)")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Test Voice Activity Detection'
    )
    
    # Test type
    test_type = parser.add_mutually_exclusive_group(required=True)
    test_type.add_argument(
        '--mic', 
        action='store_true',
        help='Test with microphone input'
    )
    test_type.add_argument(
        '--file',
        help='Test with audio file'
    )
    
    # Options
    parser.add_argument(
        '-a', '--aggressiveness',
        type=int,
        choices=[0, 1, 2, 3],
        default=3,
        help='VAD aggressiveness (0-3, default: 3)'
    )
    parser.add_argument(
        '-d', '--duration',
        type=int,
        default=10,
        help='Test duration in seconds for microphone test (default: 10)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)
    
    try:
        if args.mic:
            test_mic_vad(args.duration, args.aggressiveness)
        elif args.file:
            test_file_vad(args.file, args.aggressiveness)
            
    except KeyboardInterrupt:
        print("\nTest interrupted")
    except Exception as e:
        print(f"Error: {e}")
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main())