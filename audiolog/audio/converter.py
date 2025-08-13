import os
import sys
import time
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import ffmpeg

from audiolog.utils.config import config
from audiolog.audio.storage import AudioStorage

logger = logging.getLogger(__name__)


class AudioConverter:
    """
    Tool to convert audio files to Opus format.
    Supports WAV, MP3, FLAC, and other formats supported by ffmpeg.
    """
    
    def __init__(self, output_dir: Optional[str] = None, bitrate: str = "32k"):
        """
        Initialize the audio converter.
        
        Args:
            output_dir: Directory to save converted files (default: same as input)
            bitrate: Output bitrate for Opus encoding
        """
        self.output_dir = output_dir
        self.bitrate = bitrate
        
        # Verify ffmpeg is available
        self._check_ffmpeg()
        logger.debug(f"Initialized AudioConverter (output_dir={output_dir}, bitrate={bitrate})")
    
    def _check_ffmpeg(self) -> None:
        """Check if ffmpeg is available."""
        try:
            probe = ffmpeg.probe(None)
        except ffmpeg.Error:
            # This is expected since we're not providing a file
            pass
        except Exception as e:
            logger.error(f"FFmpeg not available: {e}")
            raise RuntimeError("FFmpeg is required but not available") from e
    
    def convert_file(self, input_path: str) -> Optional[str]:
        """
        Convert an audio file to Opus format.
        
        Args:
            input_path: Path to the input audio file
            
        Returns:
            Path to the converted file or None if conversion failed
        """
        if not os.path.exists(input_path):
            logger.error(f"Input file not found: {input_path}")
            return None
            
        try:
            # Get input file details
            input_path = os.path.abspath(input_path)
            input_dir = os.path.dirname(input_path)
            input_filename = os.path.basename(input_path)
            input_name, input_ext = os.path.splitext(input_filename)
            
            # Determine output path
            if self.output_dir:
                output_dir = os.path.abspath(self.output_dir)
                os.makedirs(output_dir, exist_ok=True)
            else:
                output_dir = input_dir
                
            output_path = os.path.join(output_dir, f"{input_name}.opus")
            
            # Skip if output file already exists
            if os.path.exists(output_path):
                logger.warning(f"Output file already exists: {output_path}")
                return output_path
                
            # Get input file details with ffprobe
            try:
                probe = ffmpeg.probe(input_path)
                audio_stream = next(
                    (stream for stream in probe['streams'] if stream['codec_type'] == 'audio'),
                    None
                )
                
                if audio_stream is None:
                    logger.error(f"No audio stream found in {input_path}")
                    return None
                    
                logger.debug(f"Input audio: {audio_stream.get('codec_name', 'unknown')} format, "
                           f"{audio_stream.get('sample_rate', 'unknown')}Hz, "
                           f"{audio_stream.get('channels', 'unknown')} channels")
                           
            except ffmpeg.Error as e:
                logger.warning(f"Could not probe file {input_path}: {e}")
                # Continue anyway, ffmpeg can still try to convert
            
            # Convert to Opus
            logger.info(f"Converting {input_path} to {output_path}")
            
            start_time = time.time()
            ffmpeg.input(input_path).output(
                output_path,
                acodec='libopus',
                audio_bitrate=self.bitrate
            ).run(quiet=True, overwrite_output=True)
            
            duration = time.time() - start_time
            
            # Verify output file exists and has non-zero size
            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                logger.error(f"Conversion failed: output file is missing or empty")
                return None
                
            logger.info(f"Successfully converted to {output_path} in {duration:.2f}s")
            return output_path
            
        except ffmpeg.Error as e:
            logger.error(f"FFmpeg error converting {input_path}: {e.stderr.decode() if e.stderr else str(e)}")
            return None
            
        except Exception as e:
            logger.error(f"Error converting {input_path}: {e}")
            return None
    
    def convert_files(self, input_paths: List[str]) -> Dict[str, str]:
        """
        Convert multiple audio files to Opus format.
        
        Args:
            input_paths: List of input file paths
            
        Returns:
            Dictionary mapping input paths to output paths for successful conversions
        """
        results = {}
        
        for input_path in input_paths:
            output_path = self.convert_file(input_path)
            if output_path:
                results[input_path] = output_path
                
        logger.info(f"Converted {len(results)}/{len(input_paths)} files")
        return results
    
    def convert_directory(self, input_dir: str, recursive: bool = False) -> Dict[str, str]:
        """
        Convert all audio files in a directory to Opus format.
        
        Args:
            input_dir: Input directory
            recursive: Whether to process subdirectories
            
        Returns:
            Dictionary mapping input paths to output paths for successful conversions
        """
        input_dir = os.path.abspath(input_dir)
        
        if not os.path.isdir(input_dir):
            logger.error(f"Input directory not found: {input_dir}")
            return {}
            
        # Find audio files
        audio_extensions = ['.wav', '.mp3', '.flac', '.aac', '.m4a', '.ogg']
        input_paths = []
        
        if recursive:
            for root, _, files in os.walk(input_dir):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in audio_extensions):
                        input_paths.append(os.path.join(root, file))
        else:
            for file in os.listdir(input_dir):
                if any(file.lower().endswith(ext) for ext in audio_extensions):
                    input_paths.append(os.path.join(input_dir, file))
        
        logger.info(f"Found {len(input_paths)} audio files in {input_dir}")
        return self.convert_files(input_paths)
    
    def save_to_storage(self, 
                        converted_files: Dict[str, str], 
                        custom_device_id: Optional[str] = None) -> List[Tuple[str, str]]:
        """
        Move converted files to the AudioLog storage structure.
        
        Args:
            converted_files: Dictionary of input to output file paths
            custom_device_id: Custom device ID for metadata
            
        Returns:
            List of tuples (opus_file_path, metadata_file_path) for saved files
        """
        storage = AudioStorage()
        results = []
        
        for input_path, opus_path in converted_files.items():
            try:
                # Get original file stats
                input_stats = os.stat(input_path)
                creation_time = input_stats.st_ctime
                modified_time = input_stats.st_mtime
                
                # Create metadata
                device_id = custom_device_id or config.get('device.id', 'default_device')
                metadata = {
                    'device_id': device_id,
                    'original_file': input_path,
                    'converted_time': time.time(),
                    'start_time': creation_time,
                    'end_time': modified_time,
                    'duration': modified_time - creation_time
                }
                
                # Save to storage
                opus_file, metadata_file = storage.save_audio_file(
                    opus_path, 
                    metadata, 
                    device_id=device_id
                )
                
                results.append((opus_file, metadata_file))
                logger.info(f"Saved to storage: {opus_file}")
                
            except Exception as e:
                logger.error(f"Error saving {opus_path} to storage: {e}")
        
        return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Convert audio files to Opus format.'
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        'input_paths', 
        nargs='*', 
        help='Paths to input audio files', 
        default=[]
    )
    input_group.add_argument(
        '-d', '--directory', 
        help='Process all audio files in a directory'
    )
    
    # Output options
    parser.add_argument(
        '-o', '--output-dir', 
        help='Directory to save converted files'
    )
    parser.add_argument(
        '-b', '--bitrate', 
        default='32k',
        help='Output bitrate (default: 32k)'
    )
    parser.add_argument(
        '-r', '--recursive', 
        action='store_true',
        help='Process subdirectories when using --directory'
    )
    
    # Storage options
    parser.add_argument(
        '-s', '--save-to-storage', 
        action='store_true',
        help='Save converted files to AudioLog storage structure'
    )
    parser.add_argument(
        '--device-id', 
        help='Custom device ID for file metadata'
    )
    
    # Misc options
    parser.add_argument(
        '-v', '--verbose', 
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )


def main():
    """Main entry point for the converter tool."""
    args = parse_args()
    setup_logging(args.verbose)
    
    try:
        # Create converter
        converter = AudioConverter(
            output_dir=args.output_dir,
            bitrate=args.bitrate
        )
        
        # Convert files
        if args.directory:
            converted_files = converter.convert_directory(
                args.directory,
                recursive=args.recursive
            )
        else:
            converted_files = converter.convert_files(args.input_paths)
        
        # Check results
        if not converted_files:
            logger.error("No files were converted successfully")
            return 1
            
        # Save to storage if requested
        if args.save_to_storage:
            saved_files = converter.save_to_storage(
                converted_files,
                custom_device_id=args.device_id
            )
            
            if saved_files:
                logger.info(f"Saved {len(saved_files)} files to storage")
            else:
                logger.warning("No files were saved to storage")
        
        logger.info(f"Successfully converted {len(converted_files)} files")
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())