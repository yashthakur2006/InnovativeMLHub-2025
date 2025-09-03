# Utility: simple metric helpers and JSON logger
import json, os, time
from dataclasses import dataclass

@dataclass
class RunMetric:
    name: str
    value: float

def save_metrics(metrics: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"metrics_{int(time.time())}.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    return path
