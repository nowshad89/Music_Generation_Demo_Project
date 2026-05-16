# main.py
import os
import sys
from src.pipeline import MusicPipeline

# Ensure output folder exists
os.makedirs("output", exist_ok=True)

MODEL_PATH = "artifacts/bach_markov.pkl"
OUTPUT_FILE = "output/bach_generated.mid"

def main():
    # Safety Check: Did you actually train a model first?
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: No trained model found at '{MODEL_PATH}'.")
        print("Please run 'python train.py' first to build the model matrix.")
        sys.exit(1)

    print("--- Launching Generation Pipeline ---")
    
    # Initialize pipeline (it will automatically find the saved order inside the pkl)
    pipeline = MusicPipeline(model_path=MODEL_PATH)
    
    # 1. Generate music tokens and save to MIDI
    pipeline.run_generation_flow(output_path=OUTPUT_FILE, length=60)
    
    # 2. Play the music immediately
    print(f"\n--- Playing: {OUTPUT_FILE} ---")
    pipeline.play_output(OUTPUT_FILE)

if __name__ == "__main__":
    main()