import json
import os

STATE_FILE = ".phaseviz_state.json" # Json file stores state

def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

def load_state() -> dict:
    if not os.path.exists(STATE_FILE):
        raise RuntimeError("No state found. Run `load-dir` first.")
    with open(STATE_FILE, "r") as f:
        return json.load(f)
