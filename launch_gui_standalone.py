#!/usr/bin/env python3

import os
import sys
import logging
import signal
import argparse
import threading
from tkinterdnd2 import TkinterDnD

# Set up logging to file instead of console
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='gui_dashboard.log',
    filemode='a'
)
logger = logging.getLogger(__name__)

# Global references to prevent garbage collection
app = None
root = None

def signal_handler(sig, frame):
    """Handle exit signals properly"""
    logger.info("Shutdown signal received, closing application")
    if root:
        root.quit()
    sys.exit(0)

def main():
    global app, root
    
    parser = argparse.ArgumentParser(description="Launch Deep Tree Echo GUI Dashboard (Standalone)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--no-activity", action="store_true", help="Disable activity monitoring")
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        
    # Set up signal handlers for graceful exit
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Import necessary components
        from gui_dashboard import GUIDashboard
        from memory_management import HypergraphMemory
        
        # Create tkinter root window
        root = TkinterDnD.Tk()
        
        # Prepare optional components with safe defaults
        memory = HypergraphMemory(storage_dir="echo_memory")
        cognitive = None
        personality = None
        sensory = None
        emergency = None
        browser = None
        ai_manager = None
        
        # Only import and initialize ActivityRegulator if not disabled
        if args.no_activity:
            activity = None
        else:
            from activity_regulation import ActivityRegulator
            activity = ActivityRegulator()
        
        # Initialize the GUI dashboard with available components
        app = GUIDashboard(
            root, 
            memory=memory,
            cognitive=cognitive,
            personality=personality,
            sensory=sensory, 
            activity=activity,
            emergency=emergency,
            browser=browser,
            ai_manager=ai_manager
        )
        
        # Configure window settings
        root.protocol("WM_DELETE_WINDOW", lambda: (root.quit(), root.destroy()))
        
        # Center window on screen
        window_width = 1200
        window_height = 800
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        center_x = int(screen_width/2 - window_width/2)
        center_y = int(screen_height/2 - window_height/2)
        root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
        
        # Set window title with version
        root.title("Deep Tree Echo Dashboard v1.0")
        
        # Log GUI ready
        logger.info("GUI dashboard initialized and ready")
        
        # Start the main event loop
        root.mainloop()
        
        # After mainloop exits
        logger.info("GUI dashboard closed")
        
        # Clean up (if activity monitoring was enabled)
        if activity:
            activity.shutdown()
        
        return 0
        
    except Exception as e:
        logger.error(f"Error launching GUI dashboard: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())