import json
import os

STATE_FILE = "warmup_state.json"

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    else:
        return {
            "enabled": False,
            "progress": 0,
            "start_time": None
        }

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)
