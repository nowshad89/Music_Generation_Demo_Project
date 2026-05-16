import os
from src.pipeline import MusicPipeline

# Ensure the folder for the model artifact exists
os.makedirs("artifacts", exist_ok=True)

# Define configuration
MARKOV_ORDER = 3  # Tweak your parameter here before training!
MODEL_PATH = "artifacts/bach_markov.pkl"

def main():
    print(f"--- Launching Training Pipeline (Order: {MARKOV_ORDER}) ---")
    
    # Initialize the pipeline with the specific order you want to test
    pipeline = MusicPipeline(model_path=MODEL_PATH, order=MARKOV_ORDER)

    # Crucial Change: Go inside your pipeline object to modify the loader's limit
    pipeline.loader.limit = 400
    
    # Run the data extraction and training logic
    pipeline.run_training_flow()
    
    print(f"SUCCESS: Model trained and saved to '{MODEL_PATH}'")

if __name__ == "__main__":
    main()