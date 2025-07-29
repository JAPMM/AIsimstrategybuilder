from backend.your_previous_model import main as full_training_loop

def train_model(shot_data=None):
    # shot_data will be passed, but your model currently loads CSVs on its own.
    # So for now we simply call your main() entrypoint.
    full_training_loop()
    return {"status": "Training complete"}
