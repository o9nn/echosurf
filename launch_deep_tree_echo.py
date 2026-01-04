#!/usr/bin/env python3

import os
import sys
import asyncio
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

# Import core modules
from deep_tree_echo import DeepTreeEcho
from memory_management import HypergraphMemory
from cognitive_architecture import CognitiveArchitecture
from personality_system import PersonalitySystem
# Use the enhanced sensory motor system for 3D spatial awareness
from sensory_motor_simple import SensoryMotorSystem
from emergency_protocols import EmergencyProtocols
from activity_regulation import ActivityRegulation
from terminal_controller import TerminalController
from ai_integration import ai_manager
from browser_interface import BrowserInterface

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("deep-tree-echo.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create directory structure if needed
for dir_path in ["activity_logs", "deep_tree_echo_profile", "ai_cache"]:
    Path(dir_path).mkdir(exist_ok=True)

class SystemStatus:
    """System status tracker"""
    INITIALIZING = "initializing"
    OPERATIONAL = "operational"
    DEGRADED = "degraded"
    EMERGENCY = "emergency"
    SHUTDOWN = "shutdown"

async def main():
    """Main entry point for Deep Tree Echo"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Launch Deep Tree Echo")
    parser.add_argument("--gui", action="store_true", help="Launch GUI dashboard")
    parser.add_argument("--browser", action="store_true", help="Initialize browser automation")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        logger.info("Initializing Deep Tree Echo...")
        status = SystemStatus.INITIALIZING
        
        # Initialize core systems
        memory = HypergraphMemory()
        cognitive = CognitiveArchitecture()
        personality = PersonalitySystem()
        sensory = SensoryMotorSystem()
        emergency = EmergencyProtocols()
        activity = ActivityRegulation()
        terminal = TerminalController()
        
        # Initialize browser interface if requested
        browser = None
        if args.browser:
            try:
                logger.info("Initializing browser automation...")
                browser = BrowserInterface()
                await browser.initialize()
                logger.info("Browser automation initialized")
            except Exception as e:
                logger.error(f"Failed to initialize browser: {str(e)}")
                emergency.register_incident("browser_init_failure", str(e))
        
        # Log successful initialization
        logger.info("Core systems initialized")
        
        # Create record of initialization experience
        init_experience = personality.create_experience(
            description="System initialization complete",
            impact=0.7,
            context={
                "event": "initialization",
                "success": True
            }
        )
        personality.process_experience(init_experience)
        cognitive.learn_from_experience(init_experience.__dict__)
        
        # Update emergency protocols state
        emergency.update_state("operational")
        emergency.update_activity()
        
        # Record successful AI service initialization
        available_ai_services = list(ai_manager.services.keys())
        if available_ai_services:
            logger.info(f"AI services available: {', '.join(available_ai_services)}")
            cognitive.store_memory(
                content=f"AI services initialized: {', '.join(available_ai_services)}",
                memory_type="declarative", 
                attributes={"source": "system", "type": "initialization"}
            )
        else:
            logger.warning("No AI services available")
            cognitive.store_memory(
                content="No AI services available",
                memory_type="declarative",
                attributes={"source": "system", "type": "warning"}
            )
        
        # Schedule periodic tasks
        async def update_models():
            logger.debug("Updating machine learning models...")
            if hasattr(cognitive, 'update_models'):
                await cognitive.update_models()
        
        async def process_sensory():
            logger.debug("Processing sensory inputs...")
            await sensory.process_all()
        
        async def check_goals():
            logger.debug("Checking goal progress...")
            if hasattr(cognitive, 'check_goals'):
                await cognitive.check_goals()
        
        async def save_state():
            logger.debug("Saving system state...")
            memory.save_state()
            cognitive.save_state()
            personality.save_state()
            
        async def run_ai_tasks():
            logger.debug("Running AI enhancement tasks...")
            
            # Get concepts from memory to enhance connections
            concepts = memory.get_all_concepts(limit=100)
            
            # Choose a random memory to analyze if available
            latest_memories = memory.get_recent_memories(memory_type="declarative", limit=5)
            if not latest_memories:
                return
                
            for memory_item in latest_memories:
                try:
                    # Analyze memory content
                    analysis = await ai_manager.analyze_text(memory_item.content, "general")
                    logger.debug(f"AI analysis complete: {analysis}")
                    
                    # Find potential connections to existing concepts
                    connections = await ai_manager.generate_echo_connections(
                        memory_item.content, [c.name for c in concepts]
                    )
                    
                    # Store connections
                    for connection in connections:
                        try:
                            memory.create_connection(
                                source=connection.get("source", ""),
                                target=connection.get("target", ""),
                                relationship=connection.get("relationship", "related_to"),
                                weight=float(connection.get("strength", 0.5)),
                                attributes={
                                    "explanation": connection.get("explanation", ""),
                                    "source": "ai_enhanced"
                                }
                            )
                        except Exception as conn_err:
                            logger.error(f"Error creating connection: {conn_err}")
                except Exception as e:
                    logger.error(f"Error in AI task: {str(e)}")
            
        # Schedule tasks with different priorities and intervals
        activity.schedule_task(
            "model_update",
            update_models,
            priority=activity.TaskPriority.LOW,
            interval=3600,  # Every hour
            cpu_threshold=0.6
        )
        
        activity.schedule_task(
            "sensory_processing",
            process_sensory,
            priority=activity.TaskPriority.HIGH,
            interval=0.1,  # Every 100ms
            cpu_threshold=0.8
        )
        
        activity.schedule_task(
            "goal_check",
            check_goals,
            priority=activity.TaskPriority.MEDIUM,
            interval=300,  # Every 5 minutes
            cpu_threshold=0.7
        )
        
        activity.schedule_task(
            "state_saving",
            save_state,
            priority=activity.TaskPriority.LOW,
            interval=900,  # Every 15 minutes
            cpu_threshold=0.5
        )
        
        activity.schedule_task(
            "ai_enhancement",
            run_ai_tasks,
            priority=activity.TaskPriority.MEDIUM,
            interval=1800,  # Every 30 minutes
            cpu_threshold=0.6
        )
        
        # Launch GUI if requested
        if args.gui:
            try:
                from gui_dashboard import launch_dashboard
                logger.info("Launching GUI dashboard...")
                # Pass our systems to the dashboard
                launch_dashboard(
                    memory=memory,
                    cognitive=cognitive,
                    personality=personality,
                    sensory=sensory,
                    activity=activity,
                    emergency=emergency,
                    browser=browser,
                    ai_manager=ai_manager  # Pass AI manager to dashboard
                )
            except Exception as e:
                logger.error(f"Failed to launch GUI dashboard: {str(e)}")
                emergency.register_incident("gui_launch_failure", str(e))
        
        # Update status
        status = SystemStatus.OPERATIONAL
        logger.info("Deep Tree Echo is now operational")
        
        # Signal operational status to terminal
        terminal.signal_status(status)
        
        # Run main event loop
        try:
            # Keep the event loop running until interrupted
            while True:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Shutdown initiated by keyboard interrupt")
            status = SystemStatus.SHUTDOWN
        
    except Exception as e:
        logger.error(f"Critical error in main routine: {str(e)}")
        status = SystemStatus.EMERGENCY
        
        # Try to log the error through emergency protocols
        try:
            emergency.register_incident("critical_failure", str(e))
        except:
            pass
    
    finally:
        # Clean shutdown
        logger.info("Shutting down...")
        
        # Close browser if initialized
        if args.browser and browser:
            try:
                await browser.close()
                logger.info("Browser automation closed")
            except Exception as e:
                logger.error(f"Error closing browser: {str(e)}")
        
        # Save final state
        try:
            memory.save_state()
            cognitive.save_state()
            personality.save_state()
            logger.info("Final state saved")
        except Exception as e:
            logger.error(f"Error saving final state: {str(e)}")
        
        # Signal shutdown to terminal
        terminal.signal_status(SystemStatus.SHUTDOWN)
        logger.info("Deep Tree Echo shutdown complete")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown initiated by keyboard interrupt")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)

