"""
Utility functions for DNA sequence processing, model training, and evaluation.
"""

import numpy as np
import pandas as pd
import os
import logging
from typing import List, Tuple, Optional, Dict, Any
from gensim.models import Word2Vec
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, 
    ReduceLROnPlateau, 
    ModelCheckpoint,
    History
)

from config import (
    DNA_ALPHABET, DNA_BASES, DEFAULT_KMER_SIZE, W2V_VECTOR_SIZE,
    W2V_MODEL_PATH, CNN_MODEL_PATH, W2V_WINDOW_SIZE, W2V_MIN_COUNT,
    W2V_EPOCHS, W2V_WORKERS, CNN_FILTERS_1, CNN_KERNEL_SIZE_1,
    CNN_FILTERS_2, CNN_KERNEL_SIZE_2, CNN_DENSE_UNITS, CNN_DROPOUT_RATE,
    LEARNING_RATE, MIN_LEARNING_RATE
)

# === Logging Configuration ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# DNA Sequence Validation and Processing
# ============================================================================

def is_valid_dna(sequence: str) -> bool:
    """
    Validate if a string is a valid DNA sequence.
    
    Args:
        sequence: String to validate
        
    Returns:
        True if valid DNA sequence, False otherwise
    """
    if not sequence or not isinstance(sequence, str):
        return False
    return set(sequence.upper()).issubset(DNA_BASES)


def sanitize_sequence(sequence: str) -> str:
    """
    Clean and normalize a DNA sequence.
    
    Args:
        sequence: Raw DNA sequence
        
    Returns:
        Uppercase, cleaned DNA sequence
    """
    return ''.join(c.upper() for c in sequence if c.upper() in DNA_BASES)


def calculate_gc_content(sequence: str) -> float:
    """
    Calculate GC content of a DNA sequence.
    
    Args:
        sequence: DNA sequence
        
    Returns:
        GC content as a fraction (0-1)
    """
    sequence = sequence.upper()
    gc_count = sequence.count('G') + sequence.count('C')
    return gc_count / len(sequence) if sequence else 0.0


def calculate_nucleotide_frequencies(sequence: str) -> Dict[str, float]:
    """
    Calculate nucleotide frequencies in a DNA sequence.
    
    Args:
        sequence: DNA sequence
        
    Returns:
        Dictionary with frequency of each nucleotide
    """
    sequence = sequence.upper()
    total = len(sequence)
    if total == 0:
        return {base: 0.0 for base in DNA_ALPHABET}
    return {base: sequence.count(base) / total for base in DNA_ALPHABET}


# ============================================================================
# K-mer Processing
# ============================================================================

def get_kmers(sequence: str, k: int = DEFAULT_KMER_SIZE) -> List[str]:
    """
    Generate k-mers from a DNA sequence.
    
    Args:
        sequence: DNA sequence
        k: K-mer size
        
    Returns:
        List of k-mers
    """
    sequence = sequence.upper()
    if len(sequence) < k:
        return [sequence]
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]


def get_all_possible_kmers(k: int = DEFAULT_KMER_SIZE) -> List[str]:
    """
    Generate all possible k-mers for a given k.
    
    Args:
        k: K-mer size
        
    Returns:
        List of all possible k-mers
    """
    if k == 1:
        return DNA_ALPHABET.copy()
    return [a + b for a in get_all_possible_kmers(k-1) for b in DNA_ALPHABET]


# ============================================================================
# Sequence Encoding
# ============================================================================

def one_hot_encode_sequences(sequences: List[str]) -> np.ndarray:
    """
    One-hot encode a list of DNA sequences.
    
    Args:
        sequences: List of DNA sequences
        
    Returns:
        numpy array of shape (n_sequences, seq_length, 4)
    """
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seq_len = len(sequences[0])
    
    one_hot = np.zeros((len(sequences), seq_len, 4), dtype=np.float32)
    
    for i, seq in enumerate(sequences):
        for j, base in enumerate(seq):
            if base in mapping:
                one_hot[i, j, mapping[base]] = 1.0
                
    return one_hot


def sequence_to_one_hot(sequence: str) -> np.ndarray:
    """
    One-hot encode a single DNA sequence.
    
    Args:
        sequence: DNA sequence
        
    Returns:
        numpy array of shape (seq_length, 4, 1)
    """
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    one_hot = np.zeros((len(sequence), 4, 1), dtype=np.float32)
    
    for j, base in enumerate(sequence.upper()):
        if base in mapping:
            one_hot[j, mapping[base], 0] = 1.0
            
    return one_hot


# ============================================================================
# Word2Vec Operations
# ============================================================================

def train_word2vec(
    sequences: List[str],
    k: int = DEFAULT_KMER_SIZE,
    vector_size: int = W2V_VECTOR_SIZE,
    window: int = W2V_WINDOW_SIZE,
    min_count: int = W2V_MIN_COUNT,
    epochs: int = W2V_EPOCHS,
    workers: int = W2V_WORKERS
) -> Word2Vec:
    """
    Train Word2Vec model on DNA k-mers.
    
    Args:
        sequences: List of DNA sequences
        k: K-mer size
        vector_size: Dimensionality of word vectors
        window: Context window size
        min_count: Minimum word count
        epochs: Number of training epochs
        workers: Number of worker threads
        
    Returns:
        Trained Word2Vec model
    """
    logger.info(f"Tokenizing sequences with k={k}...")
    tokenized = [get_kmers(seq.upper(), k) for seq in sequences]
    
    total_kmers = sum(len(t) for t in tokenized)
    unique_kmers = len(set(k for t in tokenized for k in t))
    logger.info(f"Total k-mers: {total_kmers}, Unique k-mers: {unique_kmers}")
    
    logger.info("Training Word2Vec model...")
    model = Word2Vec(
        sentences=tokenized,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=epochs
    )
    
    model.save(W2V_MODEL_PATH)
    logger.info(f"Word2Vec model saved to {W2V_MODEL_PATH}")
    
    return model


def sequence_to_vector(
    sequence: str,
    w2v_model: Word2Vec,
    k: int = DEFAULT_KMER_SIZE
) -> np.ndarray:
    """
    Convert a DNA sequence to a vector using Word2Vec embeddings.
    
    Args:
        sequence: DNA sequence
        w2v_model: Trained Word2Vec model
        k: K-mer size used during training
        
    Returns:
        Averaged vector representation of the sequence
    """
    sequence = sequence.upper()
    kmers = get_kmers(sequence, k)
    vectors = [w2v_model.wv[kmer] for kmer in kmers if kmer in w2v_model.wv]
    
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        logger.warning(f"No valid k-mers found in sequence. Returning zero vector.")
        return np.zeros(w2v_model.vector_size)


# ============================================================================
# Model Building and Loading
# ============================================================================

def build_cnn_model(input_shape: Tuple[int, int]) -> Sequential:
    """
    Build CNN model for DNA classification.
    
    Args:
        input_shape: Shape of input data (sequence_length, 4)
        
    Returns:
        Compiled Keras Sequential model
    """
    model = Sequential([
        # First convolutional block
        Conv1D(
            filters=CNN_FILTERS_1,
            kernel_size=CNN_KERNEL_SIZE_1,
            activation='relu',
            input_shape=input_shape,
            name='conv1d_1'
        ),
        BatchNormalization(name='batch_norm_1'),
        MaxPooling1D(pool_size=2, name='max_pool_1'),
        
        # Second convolutional block
        Conv1D(
            filters=CNN_FILTERS_2,
            kernel_size=CNN_KERNEL_SIZE_2,
            activation='relu',
            name='conv1d_2'
        ),
        BatchNormalization(name='batch_norm_2'),
        MaxPooling1D(pool_size=2, name='max_pool_2'),
        
        # Classification head
        Flatten(name='flatten'),
        Dense(CNN_DENSE_UNITS, activation='relu', name='dense_1'),
        Dropout(CNN_DROPOUT_RATE, name='dropout'),
        Dense(1, activation='sigmoid', name='output')
    ])
    
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC']
    )
    
    return model


def load_models(
    w2v_path: str = W2V_MODEL_PATH,
    model_path: str = CNN_MODEL_PATH
) -> Tuple[Word2Vec, Sequential]:
    """
    Load trained Word2Vec and CNN models.
    
    Args:
        w2v_path: Path to Word2Vec model
        model_path: Path to CNN model
        
    Returns:
        Tuple of (Word2Vec model, CNN model)
        
    Raises:
        FileNotFoundError: If either model file doesn't exist
    """
    if not os.path.exists(w2v_path):
        raise FileNotFoundError(
            f"Word2Vec model not found at '{w2v_path}'. "
            "Please train the Word2Vec model first."
        )
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"CNN model not found at '{model_path}'. "
            "Please train the CNN model first."
        )
    
    logger.info(f"Loading Word2Vec model from {w2v_path}...")
    w2v_model = Word2Vec.load(w2v_path)
    
    logger.info(f"Loading CNN model from {model_path}...")
    cnn_model = load_model(model_path)
    
    return w2v_model, cnn_model


def is_model_ready() -> bool:
    """
    Check if both required models are available.
    
    Returns:
        True if both models exist, False otherwise
    """
    return os.path.exists(W2V_MODEL_PATH) and os.path.exists(CNN_MODEL_PATH)


def get_model_summary() -> Optional[str]:
    """
    Get a summary string of the CNN model.
    
    Returns:
        Model summary string or None if model doesn't exist
    """
    try:
        _, model = load_models()
        summary_lines = []
        model.summary(print_fn=lambda x: summary_lines.append(x))
        return '\n'.join(summary_lines)
    except FileNotFoundError:
        return None


# ============================================================================
# Data Analysis Utilities
# ============================================================================

def analyze_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze a DNA dataset and return statistics.
    
    Args:
        df: DataFrame with 'sequence' and 'label' columns
        
    Returns:
        Dictionary with analysis results
    """
    if df.empty:
        return {}
    
    sequences = df['sequence'].values
    labels = df['label'].values
    
    # Basic statistics
    lengths = [len(seq) for seq in sequences]
    gc_contents = [calculate_gc_content(seq) for seq in sequences]
    
    # Label distribution
    label_counts = df['label'].value_counts().to_dict()
    
    # Motif presence (TATA box)
    tata_count = sum(1 for seq in sequences if 'TATA' in seq.upper())
    
    return {
        'total_samples': len(df),
        'sequence_lengths': {
            'min': min(lengths),
            'max': max(lengths),
            'mean': np.mean(lengths),
            'std': np.std(lengths)
        },
        'gc_content': {
            'min': min(gc_contents),
            'max': max(gc_contents),
            'mean': np.mean(gc_contents),
            'std': np.std(gc_contents)
        },
        'label_distribution': {
            'positive': int(label_counts.get(1, 0)),
            'negative': int(label_counts.get(0, 0)),
            'positive_ratio': label_counts.get(1, 0) / len(df)
        },
        'tata_box_present': tata_count,
        'tata_box_ratio': tata_count / len(df)
    }


def dummy_vector(vector_size: int = W2V_VECTOR_SIZE) -> np.ndarray:
    """
    Generate a dummy vector for testing purposes.
    
    Args:
        vector_size: Size of the vector
        
    Returns:
        Random numpy array
    """
    return np.random.rand(vector_size)