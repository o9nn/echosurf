#!/usr/bin/env python3

import os
import sys
import argparse
import subprocess
import logging
import time
import signal
import threading
import atexit
import psutil
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='dashboard_launcher.log',
    filemode='a'
)
logger = logging.getLogger(__name__)

# Global process trackers
gui_process = None
web_process = None
processes = []

def signal_handler(sig, frame):
    """Handle exit signals properly"""
    logger.info("Shutdown signal received, closing dashboards")
    cleanup()
    sys.exit(0)

def cleanup():
    """Clean up all processes on exit"""
    logger.info("Cleaning up processes")
    
    for proc in processes:
        try:
            if proc.poll() is None:  # Still running
                logger.info(f"Terminating process {proc.pid}")
                proc.terminate()
                # Give it some time to terminate gracefully
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    logger.warning(f"Process {proc.pid} did not terminate gracefully, forcing")
                    proc.kill()
        except Exception as e:
            logger.error(f"Error cleaning up process: {e}")

def launch_gui_dashboard(use_locale_fix=True, env=None):
    """Launch the GUI dashboard"""
    global gui_process
    
    try:
        script = "fix_locale_gui.py" if use_locale_fix else "launch_gui_standalone.py"
        logger.info(f"Launching GUI dashboard with {script}")
        
        # Start the process
        gui_process = subprocess.Popen(
            [sys.executable, script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
            env=env
        )
        processes.append(gui_process)
        
        # Start a thread to monitor the output
        threading.Thread(target=monitor_output, 
                       args=(gui_process, "GUI Dashboard"), 
                       daemon=True).start()
        
        logger.info(f"GUI dashboard started with PID: {gui_process.pid}")
        return True
    
    except Exception as e:
        logger.error(f"Error launching GUI dashboard: {e}")
        return False

def launch_web_dashboard(port=5000):
    """Launch the Web dashboard"""
    global web_process
    
    try:
        logger.info(f"Launching Web dashboard on port {port}")
        
        # Start the process
        web_process = subprocess.Popen(
            [sys.executable, "web_gui.py", "--port", str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1  # Line buffered
        )
        processes.append(web_process)
        
        # Start a thread to monitor the output
        threading.Thread(target=monitor_output, 
                       args=(web_process, "Web Dashboard"), 
                       daemon=True).start()
        
        logger.info(f"Web dashboard started with PID: {web_process.pid}")
        return True
    
    except Exception as e:
        logger.error(f"Error launching Web dashboard: {e}")
        return False

def monitor_output(process, name):
    """Monitor and log the output from a process"""
    while process.poll() is None:  # While the process is still running
        try:
            # Read output line by line
            for line in process.stdout:
                if line:
                    logger.info(f"{name} output: {line.strip()}")
            
            # Check for errors
            for line in process.stderr:
                if line:
                    logger.error(f"{name} error: {line.strip()}")
        except Exception as e:
            logger.error(f"Error reading {name} output: {e}")
            time.sleep(1)  # Prevent tight error loop
    
    exit_code = process.returncode
    logger.info(f"{name} process exited with code: {exit_code}")

def check_dashboard_status():
    """Check the status of both dashboards"""
    gui_status = "Running" if gui_process and gui_process.poll() is None else "Not running"
    web_status = "Running" if web_process and web_process.poll() is None else "Not running"
    
    # Get port information for web dashboard if it's running
    web_port = "N/A"
    web_url = "N/A"
    if web_process and web_process.poll() is None:
        try:
            for conn in psutil.Process(web_process.pid).connections():
                if conn.status == 'LISTEN':
                    web_port = conn.laddr.port
                    web_url = f"http://localhost:{web_port}"
                    break
        except Exception:
            pass
    
    return {
        "gui_status": gui_status,
        "web_status": web_status,
        "gui_pid": gui_process.pid if gui_process else None,
        "web_pid": web_process.pid if web_process else None,
        "web_port": web_port,
        "web_url": web_url
    }

def find_forwarded_ports():
    """Find potential forwarded ports for the web dashboard in container environments"""
    forwarded_ports = []
    
    # Check for common environment variables used in dev containers, Codespaces, etc.
    for key, value in os.environ.items():
        if key.startswith('FORWARDED_PORT_') or key.startswith('PORT_'):
            forwarded_ports.append(value)
    
    # Check for Codespaces URL pattern
    hostname = os.environ.get('HOSTNAME', '')
    codespace_name = os.environ.get('CODESPACE_NAME', '')
    if codespace_name and 'GITHUB_CODESPACES' in os.environ:
        ports = [5000]  # Default port
        for port in ports:
            forwarded_ports.append(f"https://{codespace_name}-{port}.preview.app.github.dev")
    
    return forwarded_ports

def print_dashboard_info(status):
    """Print dashboard status information to the console"""
    print("\n" + "="*80)
    print("DEEP TREE ECHO DASHBOARDS")
    print("="*80)
    
    print(f"\nGUI Dashboard: {status['gui_status']}")
    if status['gui_pid']:
        print(f"  - PID: {status['gui_pid']}")
    
    print(f"\nWeb Dashboard: {status['web_status']}")
    if status['web_pid']:
        print(f"  - PID: {status['web_pid']}")
        print(f"  - URL: {status['web_url']}")
        
        # Show potential forwarded ports/URLs
        forwarded = find_forwarded_ports()
        if forwarded:
            print("\nPossible forwarded URLs:")
            for url in forwarded:
                print(f"  - {url}")
    
    print("\nTo stop both dashboards, press Ctrl+C")
    print("="*80 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Launch and manage Deep Tree Echo dashboards")
    parser.add_argument("--gui-only", action="store_true", help="Launch only the GUI dashboard")
    parser.add_argument("--web-only", action="store_true", help="Launch only the Web dashboard")
    parser.add_argument("--no-locale-fix", action="store_true", 
                      help="Don't use the locale fix for the GUI dashboard")
    parser.add_argument("--web-port", type=int, default=8080, 
                      help="Port for the Web dashboard (default: 8080)")
    parser.add_argument("--gui-port", type=int, default=5000,
                      help="Port for the GUI dashboard when using the web-based version (default: 5000)")
    parser.add_argument("--no-monitor", action="store_true", 
                      help="Don't monitor dashboard status")
    
    args = parser.parse_args()
    
    # Set up signal handlers for clean exit
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Register cleanup function
    atexit.register(cleanup)
    
    # Launch dashboards based on arguments
    launched_gui = False
    launched_web = False
    
    if not args.web_only:
        # For GUI, pass the port as an environment variable since fix_locale_gui.py might not accept a port argument
        env = os.environ.copy()
        env['GUI_DASHBOARD_PORT'] = str(args.gui_port)
        launched_gui = launch_gui_dashboard(not args.no_locale_fix, env=env)
    
    if not args.gui_only:
        launched_web = launch_web_dashboard(args.web_port)
    
    if not launched_gui and not launched_web:
        logger.error("Failed to launch any dashboards")
        return 1
    
    # Monitor dashboard status
    try:
        if not args.no_monitor:
            while True:
                # Get current status
                status = check_dashboard_status()
                print_dashboard_info(status)
                
                # Exit if all processes have terminated
                if ((args.gui_only or args.web_only) and 
                    ((args.gui_only and status["gui_status"] == "Not running") or
                     (args.web_only and status["web_status"] == "Not running"))):
                    logger.info("All requested dashboards have terminated")
                    break
                elif (not args.gui_only and not args.web_only and
                     status["gui_status"] == "Not running" and
                     status["web_status"] == "Not running"):
                    logger.info("All dashboards have terminated")
                    break
                
                # Wait before checking again
                time.sleep(10)
        else:
            logger.info("Not monitoring dashboard status")
            # Just print the initial status
            status = check_dashboard_status()
            print_dashboard_info(status)
            
            # Keep the script running until Ctrl+C
            signal.pause()
    
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        cleanup()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())