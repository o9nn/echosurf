
import sys
import yaml

def validate_workflow(file_path):
    with open(file_path, 'r') as f:
        workflow = yaml.safe_load(f)

    # Example: Ensure the cron syntax is valid
    if 'on' in workflow and 'schedule' in workflow['on']:
        cron = workflow['on']['schedule'][0].get('cron', '')
        if not cron or len(cron.split()) != 5:
            raise ValueError(f"Invalid cron syntax: {cron}")

if __name__ == "__main__":
    file_path = sys.argv[1]
    try:
        validate_workflow(file_path)
        print("Validation passed.")
    except Exception as e:
        print(f"Validation failed: {e}")
        sys.exit(1)
