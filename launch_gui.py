#!/usr/bin/env python3

import os
import sys
import traceback
import logging
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure X11 display is available
def ensure_display():
    """Check and set up display environment if needed"""
    if 'DISPLAY' not in os.environ:
        logger.error("No display environment detected. GUI requires X11 display.")
        return False
        
    # Try to access the X server to verify connection
    try:
        import tkinter as tk
        test_window = tk.Tk()
        test_window.withdraw()  # Hide the window
        display_info = f"Display connected at {os.environ.get('DISPLAY')} with geometry {test_window.winfo_screenwidth()}x{test_window.winfo_screenheight()}"
        logger.info(display_info)
        test_window.destroy()
        return True
    except Exception as e:
        logger.error(f"Display error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Launch Deep Tree Echo GUI Dashboard")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Print environment info
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Display environment: {os.environ.get('DISPLAY', 'Not set')}")
    logger.info(f"Current directory: {os.getcwd()}")
    
    # Check for display
    if not ensure_display():
        logger.error("Cannot start GUI: X11 display not available")
        return 1
    
    try:
        logger.info("Attempting to import tkinterdnd2")
        from tkinterdnd2 import TkinterDnD
        logger.info("Successfully imported tkinterdnd2")
        
        logger.info("Importing components...")
        
        try:
            logger.info("Importing GUIDashboard")
            from gui_dashboard import GUIDashboard
            logger.info("Imported GUIDashboard successfully")
        except Exception as e:
            logger.error(f"Failed to import GUIDashboard: {e}")
            logger.error(traceback.format_exc())
            return 1
            
        try:
            logger.info("Importing HypergraphMemory")
            from memory_management import HypergraphMemory
            logger.info("Imported HypergraphMemory successfully")
        except Exception as e:
            logger.error(f"Failed to import HypergraphMemory: {e}")
            logger.error(traceback.format_exc())
            return 1
            
        try:
            logger.info("Importing CognitiveArchitecture")
            from cognitive_architecture import CognitiveArchitecture
            logger.info("Imported CognitiveArchitecture successfully")
        except Exception as e:
            logger.error(f"Failed to import CognitiveArchitecture: {e}")
            logger.error(traceback.format_exc())
            CognitiveArchitecture = lambda: None
            
        try:
            logger.info("Importing PersonalitySystem")
            from personality_system import PersonalitySystem
            logger.info("Imported PersonalitySystem successfully")
        except Exception as e:
            logger.error(f"Failed to import PersonalitySystem: {e}")
            logger.error(traceback.format_exc())
            PersonalitySystem = lambda: None
            
        try:
            logger.info("Importing SensoryMotorSystem")
            from sensory_motor_simple import SensoryMotorSystem
            logger.info("Imported SensoryMotorSystem successfully")
        except Exception as e:
            logger.error(f"Failed to import SensoryMotorSystem from sensory_motor_simple: {e}")
            logger.error(traceback.format_exc())
            try:
                logger.info("Trying to import SensoryMotorSystem from sensory_motor")
                from sensory_motor import SensoryMotorSystem
                logger.info("Imported SensoryMotorSystem from sensory_motor successfully")
            except Exception as e2:
                logger.error(f"Failed to import SensoryMotorSystem from sensory_motor: {e2}")
                logger.error(traceback.format_exc())
                SensoryMotorSystem = None
            
        try:
            logger.info("Importing EmergencyProtocols")
            from emergency_protocols import EmergencyProtocols
            logger.info("Imported EmergencyProtocols successfully")
        except Exception as e:
            logger.error(f"Failed to import EmergencyProtocols: {e}")
            logger.error(traceback.format_exc())
            EmergencyProtocols = lambda: None
            
        try:
            logger.info("Importing ActivityRegulator")
            from activity_regulation import ActivityRegulator
            logger.info("Imported ActivityRegulator successfully")
        except Exception as e:
            logger.error(f"Failed to import ActivityRegulator: {e}")
            logger.error(traceback.format_exc())
            return 1
        
        # Check for ai_integration.py
        try:
            logger.info("Importing ai_manager")
            from ai_integration import ai_manager
            logger.info("AI integration manager loaded")
        except Exception as e:
            logger.warning(f"AI integration manager not available: {e}")
            logger.error(traceback.format_exc())
            ai_manager = None
        
        # Initialize core systems
        logger.info("Initializing core systems...")
        
        logger.info("Initializing HypergraphMemory")
        try:
            memory = HypergraphMemory(storage_dir="echo_memory")
            logger.info("HypergraphMemory initialized")
        except Exception as e:
            logger.error(f"Failed to initialize HypergraphMemory: {e}")
            logger.error(traceback.format_exc())
            return 1
        
        logger.info("Initializing CognitiveArchitecture")
        try:
            cognitive = CognitiveArchitecture()
            logger.info("CognitiveArchitecture initialized")
        except Exception as e:
            logger.error(f"Failed to initialize CognitiveArchitecture: {e}")
            cognitive = None
        
        logger.info("Initializing PersonalitySystem")
        try:
            personality = PersonalitySystem()
            logger.info("PersonalitySystem initialized")
        except Exception as e:
            logger.error(f"Failed to initialize PersonalitySystem: {e}")
            personality = None
        
        # Initialize sensory system with enhanced 3D capabilities
        sensory = None
        if SensoryMotorSystem:
            logger.info("Initializing SensoryMotorSystem")
            try:
                sensory = SensoryMotorSystem()
                logger.info("Sensory motor system loaded successfully")
            except Exception as e:
                logger.error(f"Failed to initialize sensory motor system: {e}")
                logger.error(traceback.format_exc())
                sensory = None
        
        logger.info("Initializing EmergencyProtocols")
        try:
            emergency = EmergencyProtocols() if EmergencyProtocols != (lambda: None) else None
            logger.info("EmergencyProtocols initialized")
        except Exception as e:
            logger.error(f"Failed to initialize EmergencyProtocols: {e}")
            emergency = None
        
        logger.info("Initializing ActivityRegulator")
        try:
            activity = ActivityRegulator()
            logger.info("ActivityRegulator initialized")
        except Exception as e:
            logger.error(f"Failed to initialize ActivityRegulator: {e}")
            logger.error(traceback.format_exc())
            return 1
        
        logger.info("Starting GUI dashboard...")
        try:
            root = TkinterDnD.Tk()
            logger.info("TkinterDnD.Tk created successfully")
            
            logger.info("Initializing GUIDashboard")
            app = GUIDashboard(
                root, 
                memory=memory,
                cognitive=cognitive,
                personality=personality,
                sensory=sensory,
                activity=activity,
                emergency=emergency,
                browser=None,  # Browser is initialized separately if needed
                ai_manager=ai_manager
            )
            logger.info("GUIDashboard initialized successfully")
            
            # Configure window settings
            root.protocol("WM_DELETE_WINDOW", lambda: (root.quit(), root.destroy()))
            
            # Get window dimensions
            w, h = root.winfo_width(), root.winfo_height()
            logger.info(f"Dashboard window dimensions: {w}x{h}")
            
            # Start main event loop
            logger.info("GUI dashboard ready - starting mainloop")
            root.mainloop()
            logger.info("GUI mainloop exited")
            
        except Exception as e:
            logger.error(f"Failed to create GUI window: {e}")
            logger.error(traceback.format_exc())
            return 1
            
        return 0
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())