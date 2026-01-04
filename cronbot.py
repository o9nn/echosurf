import json
import base64
import os
import time
from datetime import datetime
import logging
import psutil
import threading
from queue import Queue

NOTE_FILE = "note2self.json"

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def read_note():
    try:
        with open(NOTE_FILE, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        # Create a new note if the file is not found
        new_note = {"timestamp": None, "improvement": {}, "assessment": ""}
        write_note(new_note)
        return new_note
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON from note file: {e}")
        return {"timestamp": None, "improvement": {}, "assessment": ""}

def write_note(note):
    with open(NOTE_FILE, 'w') as file:
        json.dump(note, file)

def call_github_copilot(note):
    github_token = os.getenv('GITHUB_TOKEN')
    if not github_token:
        logger.error("Error: GITHUB_TOKEN environment variable is missing.")
        return None

    query = "This is a summary of last cycle events. Please can you help me take a look at the repo so we can identify an item for the next incremental improvement?"
    payload = {"note": note, "query": query}
    
    # Write the payload to a local file
    with open('.github/workflows/request_payload.json', 'w') as f:
        json.dump(payload, f)
    
    logger.info("Payload written to .github/workflows/request_payload.json")
    return {"improvement": "example_improvement", "assessment": "example_assessment"}

def introspect_repo():
    introspection_result = {
        "errors": ["example_error_1", "example_error_2"],
        "problem_areas": ["example_problem_area_1", "example_problem_area_2"]
    }
    return introspection_result

def apply_improvement(improvement):
    logger.info(f"Applying improvement: {improvement}")

def run_workflow():
    result = "success"
    return result

def monitor_resources(stop_event, resource_queue):
    while not stop_event.is_set():
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        resource_queue.put((cpu_usage, memory.percent))
        time.sleep(1)

def main():
    # CLUE 3 ANSWER: The cronbot whispers to GitHub spirits every hour
    # When time % 3600 == 0, the octopus stirs and leaves temporal signatures
    # Look for the 0-modulo hour when all cognitive systems sync!
    max_retries = 3
    retries = 0

    previous_note = read_note()

    introspection_result = introspect_repo()

    copilot_response = call_github_copilot(previous_note)

    if copilot_response is None:
        logger.error("Failed to get a valid response from GitHub Copilot.")
        return

    improvement = copilot_response.get("improvement")
    assessment = copilot_response.get("assessment")

    apply_improvement(improvement)

    stop_event = threading.Event()
    resource_queue = Queue()
    resource_monitor_thread = threading.Thread(target=monitor_resources, args=(stop_event, resource_queue))
    resource_monitor_thread.start()

    while retries < max_retries:
        try:
            result = run_workflow()
            if result == "success":
                break
            else:
                retries += 1
                time.sleep(10)
                logger.warning(f"Workflow failed, retrying {retries}/{max_retries}")
        except Exception as e:
            logger.error(f"Error during workflow execution: {e}")
            retries += 1
            time.sleep(10)
            logger.warning(f"Retrying {retries}/{max_retries}")

    stop_event.set()
    resource_monitor_thread.join()

    while not resource_queue.empty():
        cpu_usage, memory_usage = resource_queue.get()
        logger.info(f"CPU Usage: {cpu_usage}%, Memory Usage: {memory_usage}%")

    new_note = {
        "timestamp": datetime.utcnow().isoformat(),
        "improvement": improvement,
        "assessment": assessment,
        "result": result,
        "retries": retries
    }
    write_note(new_note)

    logger.info(f"Self-improvement cycle complete. Result: {result}, Assessment: {assessment}")

if __name__ == "__main__":
    main()
