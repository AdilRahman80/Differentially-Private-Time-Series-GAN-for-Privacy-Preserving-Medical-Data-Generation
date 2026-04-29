import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset"
RESULTS_DIR = BASE_DIR / "results"
DB_PATH = BASE_DIR / "dashboard" / "project_data.db"

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DB_PATH.parent, exist_ok=True)

# Data Parameters
SEQ_LEN = 24  # Time steps per sequence (e.g., hours in a day)
FEATURE_DIM = 4  # Number of features (e.g., HR, BP, SpO2, Temp)
NUM_SAMPLES = 5000  # Number of sequences to generate initially

# Model Hyperparameters
HIDDEN_DIM = 64
NUM_LAYERS = 3
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EPOCHS = 100

# Differential Privacy Parameters
NOISE_MULTIPLIER = 1.1
MAX_GRAD_NORM = 1.0
TARGET_EPSILON = 1.0
TARGET_DELTA = 1e-5

# Features List for synthetic data
FEATURE_NAMES = ["HeartRate", "SystolicBP", "SpO2", "Temperature"]
