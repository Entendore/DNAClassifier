"""
Configuration constants for the DNA Classifier application.
Centralizes all configurable parameters for easy maintenance.
"""

import os

# === Application Settings ===
APP_TITLE = "DNA Sequence Classifier"
APP_ICON = "🧬"
APP_LAYOUT = "wide"
PAGE_TITLE = "DNA Classifier | Bioinformatics Dashboard"

# === File Paths ===
DATA_DIR = "data"
MODEL_DIR = "models"
SYNTHETIC_DATA_PATH = os.path.join(DATA_DIR, "synthetic_data.csv")
W2V_MODEL_PATH = os.path.join(MODEL_DIR, "w2v.model")
CNN_MODEL_PATH = os.path.join(MODEL_DIR, "dna_cnn_model.keras")
HISTORY_PATH = os.path.join(MODEL_DIR, "history.json")
TRAINING_LOG_PATH = os.path.join(MODEL_DIR, "training.log")

# === DNA Constants ===
DNA_ALPHABET = ['A', 'T', 'G', 'C']
DNA_BASES = set(DNA_ALPHABET)
VALID_KMER_SIZES = [3, 4, 5, 6]
DEFAULT_KMER_SIZE = 3

# === TATA Box Motif Variants ===
TATA_MOTIFS = [
    "TATAAA",    # Canonical TATA box
    "TATATA",    # Variant
    "TATAA",     # Short variant
    "CATAAA",    # C-rich variant
    "TATATAA",   # Extended variant
]

# === Synthetic Data Generation ===
DEFAULT_NUM_SAMPLES = 1000
DEFAULT_SEQ_LENGTH = 100
MIN_SEQ_LENGTH = 50
MAX_SEQ_LENGTH = 500
MOTIF_INSERTION_PROBABILITY = 0.5
NOISE_PROBABILITY = 0.02  # Probability of random base mutation

# === Word2Vec Settings ===
W2V_VECTOR_SIZE = 100
W2V_WINDOW_SIZE = 5
W2V_MIN_COUNT = 1
W2V_EPOCHS = 50
W2V_WORKERS = 4

# === CNN Model Architecture ===
CNN_FILTERS_1 = 64
CNN_KERNEL_SIZE_1 = 8
CNN_FILTERS_2 = 128
CNN_KERNEL_SIZE_2 = 8
CNN_DENSE_UNITS = 64
CNN_DROPOUT_RATE = 0.5

# === Training Settings ===
TRAINING_EPOCHS = 1000
TRAINING_BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 15
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5
MIN_LEARNING_RATE = 1e-6

# === Classification Threshold ===
CLASSIFICATION_THRESHOLD = 0.5

# === UI Theme Colors ===
COLORS = {
    "primary": "#1E88E5",
    "success": "#43A047",
    "warning": "#FB8C00",
    "danger": "#E53935",
    "info": "#00ACC1",
    "background": "#FAFAFA",
    "card_background": "#FFFFFF",
    "text_primary": "#212121",
    "text_secondary": "#757575",
}

# === Initialize Directories ===
def initialize_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)