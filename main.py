# main.py
import os
import sys

# Import your settings from the new config module
from src.config import MODEL_PATH, OUTPUT_FILE, GENERATION_LENGTH, OUTPUT_DIR
from src.pipeline import MusicPipeline

# Ensure output folder exists dynamically using config values
os.makedirs(OUTPUT_DIR, exist_ok=True)

# main.py
import os
import sys
from datetime import datetime  # <-- Import the datetime module

# Import your settings from the config module
from src.config import MODEL_PATH, OUTPUT_FILE, GENERATION_LENGTH, OUTPUT_DIR
from src.pipeline import MusicPipeline

# Ensure output folder exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    # Safety Check
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: No trained model found at '{MODEL_PATH}'.")
        print("Please run 'python train.py' first to build the model matrix.")
        sys.exit(1)

    print("--- Launching Generation Pipeline ---")
    pipeline = MusicPipeline(model_path=MODEL_PATH)
    
    # -----------------------------------------------------------------
    # AUTOMATION FIX: Create a unique timestamped filename
    # This turns 'bach_generated.mid' into something like 'gen_20261024_143022.mid'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_output_file = os.path.join(OUTPUT_DIR, f"gen_{timestamp}.mid")
    # -----------------------------------------------------------------
    
    # 1. Generate music tokens using the unique filename
    pipeline.run_generation_flow(output_path=unique_output_file, length=GENERATION_LENGTH)
    
    # 2. Play the unique music file immediately
    print(f"\n--- Playing: {unique_output_file} ---")
    pipeline.play_output(unique_output_file)

if __name__ == "__main__":
    main()