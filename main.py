# main.py
import os
import sys

# Import your settings from the new config module
from src.config import MODEL_PATH, OUTPUT_FILE, GENERATION_LENGTH, OUTPUT_DIR
from src.pipeline import MusicPipeline

# Ensure output folder exists dynamically using config values
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    # Safety Check: Did you actually train a model first?
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: No trained model found at '{MODEL_PATH}'.")
        print("Please run 'python train.py' first to build the model matrix.")
        sys.exit(1)

    print("--- Launching Generation Pipeline ---")
    
    # Initialize pipeline with the centralized model path
    pipeline = MusicPipeline(model_path=MODEL_PATH)
    
    # 1. Generate music tokens using config settings
    pipeline.run_generation_flow(output_path=OUTPUT_FILE, length=GENERATION_LENGTH)
    
    # 2. Play the music immediately
    print(f"\n--- Playing: {OUTPUT_FILE} ---")
    pipeline.play_output(OUTPUT_FILE)

if __name__ == "__main__":
    main()