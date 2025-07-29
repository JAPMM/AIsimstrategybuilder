from backend.your_previous_model import main as full_training_loop
import pandas as pd

def train_model(shot_data=None):
    # Save uploaded shot data into the same format your agent expects
    if shot_data:
        df = pd.DataFrame(shot_data)
        df.to_csv("backend/shot_logs_all.csv", index=False)

    # Now run full training loop (your model reads shot_logs_all.csv as pretraining)
    full_training_loop()
    return {"status": "Training complete (seeded with replay buffer)"}

import json
import os 

def export_strategy_to_json(stroke_zones, hole_number=1):
    export_path = f"../frontend/public/strategies/hole_{hole_number}_strategy.json
    with open(export_path, "w") as f:
        json.dump({"stroke_zones": stroke_zones }, f, indent=2)"