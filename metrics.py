import json
import pandas as pd

def load_classification_metrics():
    with open("metrics/classification_metrics.json") as f:
        return pd.DataFrame(json.load(f))

def load_yolo_metrics():
    with open("metrics/yolo_metrics.json") as f:
        return json.load(f)