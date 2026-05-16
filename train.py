# train.py
import os

# 1. CRITICAL: Import your centralized settings from the config file
from src.config import MODEL_DIR, MODEL_PATH, MARKOV_ORDER, CORPUS_LIMIT
from src.pipeline import MusicPipeline

# Ensure the folder for the model artifact exists using config values
os.makedirs(MODEL_DIR, exist_ok=True)

def main():
    # This will now dynamically read the correct order (e.g., 4) from src/config.py
    print(f"--- Launching Training Pipeline (Order: {MARKOV_ORDER}) ---")
    
    # Initialize the pipeline with parameters directly from config
    pipeline = MusicPipeline(model_path=MODEL_PATH, order=MARKOV_ORDER)

    # Automatically set the dataset limit from your config file
    pipeline.loader.limit = CORPUS_LIMIT
    
    # Run the data extraction and training logic
    pipeline.run_training_flow()
    
    print(f"SUCCESS: Model trained on {CORPUS_LIMIT} files and saved to '{MODEL_PATH}'")

if __name__ == "__main__":
    main()