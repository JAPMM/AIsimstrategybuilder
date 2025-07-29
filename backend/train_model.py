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
