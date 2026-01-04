import json
import requests
import os
import logging
import time

GITHUB_COPILOT_API_URL = "https://api.github.com/copilot/suggestions"
NOTE_FILE = "note2self.json"

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def fetch_suggestions_from_copilot(note):
    github_token = os.getenv('GITHUB_TOKEN')
    if not github_token:
        logger.error("Error: GITHUB_TOKEN environment variable is missing.")
        return None

    headers = {
        "Authorization": f"Bearer {github_token}",
        "Content-Type": "application/json"
    }
    payload = {
        "note": note,
        "query": "This is a summary of last cycle events. Please can you help me take a look at the repo so we can identify an item for the next incremental improvement?"
    }

    max_retries = 3
    retries = 0
    while retries < max_retries:
        try:
            response = requests.post(GITHUB_COPILOT_API_URL, headers=headers, json=payload)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 401:
                logger.error("Error: Unauthorized. Please check your GITHUB_TOKEN.")
                return None
            else:
                logger.error(f"Failed to fetch suggestions from GitHub Copilot: {response.status_code}")
                retries += 1
                time.sleep(5)
        except requests.RequestException as e:
            logger.error(f"Request exception: {e}")
            retries += 1
            time.sleep(5)
    return None

def update_note_with_suggestions(suggestions):
    try:
        with open(NOTE_FILE, 'r') as file:
            note = json.load(file)
    except FileNotFoundError:
        note = {"timestamp": None, "improvement": {}, "assessment": ""}
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON from note file: {e}")
        note = {"timestamp": None, "improvement": {}, "assessment": ""}

    note.update(suggestions)

    with open(NOTE_FILE, 'w') as file:
        json.dump(note, file)

def main():
    try:
        with open(NOTE_FILE, 'r') as file:
            note = json.load(file)
    except FileNotFoundError:
        note = {"timestamp": None, "improvement": {}, "assessment": ""}
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON from note file: {e}")
        note = {"timestamp": None, "improvement": {}, "assessment": ""}

    suggestions = fetch_suggestions_from_copilot(note)
    if suggestions:
        update_note_with_suggestions(suggestions)
        logger.info("Note updated with suggestions from GitHub Copilot.")
    else:
        logger.error("No suggestions received from GitHub Copilot.")

if __name__ == "__main__":
    main()
