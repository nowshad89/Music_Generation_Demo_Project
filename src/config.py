# src/config.py
import os

# Hyperparameters
MARKOV_ORDER = 3
CORPUS_LIMIT = 400
GENERATION_LENGTH = 200

# File Paths
MODEL_DIR = "artifacts"
OUTPUT_DIR = "output"

MODEL_PATH = os.path.join(MODEL_DIR, "bach_markov.pkl")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "bach_generated.mid")