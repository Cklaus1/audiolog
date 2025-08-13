import os
import threading
import tempfile
import logging
from typing import Dict, Any, Callable, Optional
from pathlib import Path
import pystray
from PIL import Image, ImageDraw

from audiolog.utils.config import config

logger = logging.getLogger(__name__)

class SystemTrayIcon:
    """
    System tray icon with menu options for controlling the application.
    Supports Windows and macOS.
    """
    
    def __init__(self, 
                 title: str = "AudioLog",
                 icon_size: int = 64,
                 recording_status_callback: Optional[Callable[[], bool]] = None,
                 toggle_callback: Optional[Callable[[bool], None]] = None):
        """
        Initialize the system tray icon.
        
        Args:
            title: Title displayed on hover
            icon_size: Size of the icon in pixels
            recording_status_callback: Callback to get current recording status
            toggle_callback: Callback to toggle recording status
        """
        self.title = title
        self.icon_size = icon_size
        self.recording_status_callback = recording_status_callback or (lambda: True)
        self.toggle_callback = toggle_callback or (lambda status: None)
        self.is_recording = True
        self.icon = None
        self.tray_thread = None
        
        logger.debug(f"Initialized SystemTrayIcon (title={title})")
    
    def create_icon(self, recording: bool = True) -> Image.Image:
        """
        Create an icon image based on recording status.
        
        Args:
            recording: Whether recording is active
            
        Returns:
            PIL Image object
        """
        # Create a blank image
        icon = Image.new('RGBA', (self.icon_size, self.icon_size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(icon)
        
        # Draw a circle with different colors based on status
        padding = int(self.icon_size * 0.1)
        diameter = self.icon_size - (2 * padding)
        
        if recording:
            # Red circle for recording
            color = (255, 50, 50, 255)
        else:
            # Gray circle for paused
            color = (150, 150, 150, 255)
        
        # Draw the circle
        draw.ellipse(
            [(padding, padding), (padding + diameter, padding + diameter)],
            fill=color
        )
        
        # Add a small dot in the center if not recording
        if not recording:
            center_dot_size = int(diameter * 0.3)
            center_x = (self.icon_size - center_dot_size) // 2
            center_y = (self.icon_size - center_dot_size) // 2
            draw.ellipse(
                [(center_x, center_y), 
                 (center_x + center_dot_size, center_y + center_dot_size)],
                fill=(50, 50, 50, 255)
            )
            
        return icon
    
    def get_menu_items(self) -> list:
        """
        Get the menu items based on current status.
        
        Returns:
            List of pystray menu items
        """
        return [
            pystray.MenuItem(
                "Recording" if self.is_recording else "Paused",
                None,
                enabled=False
            ),
            pystray.MenuItem(
                "Pause Recording" if self.is_recording else "Resume Recording",
                self.on_toggle
            ),
            pystray.MenuItem(
                "Exit",
                self.on_exit
            )
        ]
    
    def create_menu(self) -> pystray.Menu:
        """
        Create the tray icon menu.
        
        Returns:
            pystray.Menu object
        """
        return pystray.Menu(*self.get_menu_items())
    
    def update_icon(self) -> None:
        """Update the icon based on current status."""
        if self.icon:
            try:
                # Get current status
                self.is_recording = self.recording_status_callback()
                
                # Update the icon
                self.icon.icon = self.create_icon(self.is_recording)
                
                # Update the menu
                self.icon.menu = self.create_menu()
                
                logger.debug(f"Updated system tray icon (recording={self.is_recording})")
            except Exception as e:
                logger.error(f"Error updating system tray icon: {e}")
    
    def on_toggle(self, icon, item) -> None:
        """
        Handle toggle menu item click.
        
        Args:
            icon: The pystray icon
            item: The clicked menu item
        """
        try:
            # Toggle recording status
            self.is_recording = not self.is_recording
            
            # Call the callback
            if self.toggle_callback:
                self.toggle_callback(self.is_recording)
                
            # Update the icon
            self.update_icon()
            
            logger.info(f"Recording {'resumed' if self.is_recording else 'paused'}")
        except Exception as e:
            logger.error(f"Error toggling recording state: {e}")
    
    def on_exit(self, icon, item) -> None:
        """
        Handle exit menu item click.
        
        Args:
            icon: The pystray icon
            item: The clicked menu item
        """
        logger.info("Exiting application from system tray")
        icon.stop()
    
    def run_tray_icon(self) -> None:
        """Run the system tray icon in its own thread."""
        try:
            # Create the icon
            self.is_recording = self.recording_status_callback()
            icon_image = self.create_icon(self.is_recording)
            
            # Create the icon
            self.icon = pystray.Icon(
                "audiolog",
                icon_image,
                self.title,
                menu=self.create_menu()
            )
            
            # Run the icon
            self.icon.run()
        except Exception as e:
            logger.error(f"Error running system tray icon: {e}")
    
    def start(self) -> None:
        """Start the system tray icon in a background thread."""
        if self.tray_thread is None or not self.tray_thread.is_alive():
            self.tray_thread = threading.Thread(
                target=self.run_tray_icon,
                daemon=True
            )
            self.tray_thread.start()
            logger.info("Started system tray icon")
    
    def stop(self) -> None:
        """Stop the system tray icon."""
        if self.icon:
            self.icon.stop()
            self.icon = None
            logger.info("Stopped system tray icon")


# Helper function to create a system tray controller
def create_tray_controller(
    recording_status_callback: Callable[[], bool],
    toggle_callback: Callable[[bool], None]
) -> SystemTrayIcon:
    """
    Create and start a system tray controller.
    
    Args:
        recording_status_callback: Callback to get current recording status
        toggle_callback: Callback to toggle recording status
        
    Returns:
        Running SystemTrayIcon instance
    """
    tray = SystemTrayIcon(
        recording_status_callback=recording_status_callback,
        toggle_callback=toggle_callback
    )
    tray.start()
    return tray