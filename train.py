"""
DNA Sequence Classification Model Training Script

Trains a CNN model on one-hot encoded DNA sequences to classify
the presence of TATA box promoter motifs.

Usage:
    python train.py --epochs 1000 --batch_size 32
"""

import argparse
import os
import time
import json
import logging
from typing import Dict, Any, Tuple
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
    Callback
)

from config import (
    SYNTHETIC_DATA_PATH, CNN_MODEL_PATH, HISTORY_PATH, TRAINING_LOG_PATH,
    TRAINING_EPOCHS, TRAINING_BATCH_SIZE, VALIDATION_SPLIT, RANDOM_SEED,
    EARLY_STOPPING_PATIENCE, REDUCE_LR_PATIENCE, REDUCE_LR_FACTOR,
    MIN_LEARNING_RATE, LEARNING_RATE
)
from utils import one_hot_encode_sequences, build_cnn_model, analyze_dataset

# === Logging Setup ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(TRAINING_LOG_PATH)
    ]
)
logger = logging.getLogger(__name__)


class TrainingLogger(Callback):
    """Custom callback for detailed training logging."""
    
    def on_train_begin(self, logs=None):
        logger.info("=" * 60)
        logger.info("Training Started")
        logger.info("=" * 60)
        logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"Epoch {epoch + 1}: "
                f"loss={logs.get('loss', 0):.4f}, "
                f"accuracy={logs.get('accuracy', 0):.4f}, "
                f"val_loss={logs.get('val_loss', 0):.4f}, "
                f"val_accuracy={logs.get('val_accuracy', 0):.4f}"
            )
    
    def on_train_end(self, logs=None):
        logger.info("=" * 60)
        logger.info("Training Completed")
        logger.info("=" * 60)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train CNN model for DNA sequence classification.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '-e', '--epochs',
        type=int,
        default=TRAINING_EPOCHS,
        help='Maximum number of training epochs'
    )
    
    parser.add_argument(
        '-b', '--batch_size',
        type=int,
        default=TRAINING_BATCH_SIZE,
        help='Training batch size'
    )
    
    parser.add_argument(
        '-v', '--validation_split',
        type=float,
        default=VALIDATION_SPLIT,
        help='Fraction of data used for validation'
    )
    
    parser.add_argument(
        '-s', '--seed',
        type=int,
        default=RANDOM_SEED,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '-d', '--data',
        type=str,
        default=SYNTHETIC_DATA_PATH,
        help='Path to training data CSV'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=CNN_MODEL_PATH,
        help='Path to save trained model'
    )
    
    parser.add_argument(
        '--verbose',
        type=int,
        default=1,
        choices=[0, 1, 2],
        help='Verbosity level (0=silent, 1=progress bar, 2=one line per epoch)'
    )
    
    return parser.parse_args()


def load_and_prepare_data(
    data_path: str,
    val_split: float,
    seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and prepare data for training.
    
    Args:
        data_path: Path to CSV file
        val_split: Validation split ratio
        seed: Random seed
        
    Returns:
        Tuple of (X_train, X_val, y_train, y_val)
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Data file not found: '{data_path}'. "
            "Please generate synthetic data first using generatesynthdata.py"
        )
    
    logger.info(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    # Basic validation
    if 'sequence' not in df.columns or 'label' not in df.columns:
        raise ValueError("Data file must contain 'sequence' and 'label' columns")
    
    # Analyze dataset
    stats = analyze_dataset(df)
    logger.info(f"Dataset statistics:")
    logger.info(f"  - Total samples: {stats['total_samples']}")
    logger.info(f"  - Positive: {stats['label_distribution']['positive']}")
    logger.info(f"  - Negative: {stats['label_distribution']['negative']}")
    logger.info(f"  - Sequence length: {stats['sequence_lengths']['mean']:.0f} ± {stats['sequence_lengths']['std']:.0f}")
    logger.info(f"  - GC content: {stats['gc_content']['mean']:.2%}")
    
    sequences = df['sequence'].values
    labels = df['label'].values
    
    # One-hot encode sequences
    logger.info("One-hot encoding sequences...")
    X = one_hot_encode_sequences(sequences)
    y = labels.astype(np.float32)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=val_split,
        random_state=seed,
        stratify=y
    )
    
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Validation samples: {len(X_val)}")
    
    return X_train, X_val, y_train, y_val


def evaluate_model(
    model: tf.keras.Model,
    X: np.ndarray,
    y: np.ndarray,
    dataset_name: str = "Validation"
) -> Dict[str, float]:
    """
    Evaluate model performance on a dataset.
    
    Args:
        model: Trained Keras model
        X: Input data
        y: True labels
        dataset_name: Name of the dataset for logging
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Get predictions
    y_pred_prob = model.predict(X, verbose=0).flatten()
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1_score': f1_score(y, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y, y_pred_prob)
    }
    
    # Log results
    logger.info(f"\n{dataset_name} Set Evaluation:")
    logger.info("-" * 40)
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    logger.info(f"\nConfusion Matrix ({dataset_name}):")
    logger.info(f"  True Negatives:  {cm[0, 0]}")
    logger.info(f"  False Positives: {cm[0, 1]}")
    logger.info(f"  False Negatives: {cm[1, 0]}")
    logger.info(f"  True Positives:  {cm[1, 1]}")
    
    return metrics


def save_training_history(
    history: Dict[str, Any],
    metrics: Dict[str, Dict[str, float]],
    output_path: str
) -> None:
    """
    Save training history and evaluation metrics to JSON.
    
    Args:
        history: Keras training history
        metrics: Evaluation metrics dictionary
        output_path: Path to save JSON file
    """
    # Convert numpy arrays to lists for JSON serialization
    history_serializable = {
        k: [float(v) for v in vals] 
        for k, vals in history.items()
    }
    
    # Add metadata
    output = {
        'training_history': history_serializable,
        'evaluation_metrics': metrics,
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'final_train_accuracy': float(history['accuracy'][-1]),
            'final_val_accuracy': float(history['val_accuracy'][-1]),
            'best_val_accuracy': float(max(history['val_accuracy'])),
            'final_train_loss': float(history['loss'][-1]),
            'final_val_loss': float(history['val_loss'][-1]),
            'best_val_loss': float(min(history['val_loss'])),
            'total_epochs': len(history['accuracy'])
        }
    }
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    logger.info(f"Training history saved to: {output_path}")


def train_model(
    epochs: int = TRAINING_EPOCHS,
    batch_size: int = TRAINING_BATCH_SIZE,
    val_split: float = VALIDATION_SPLIT,
    seed: int = RANDOM_SEED,
    data_path: str = SYNTHETIC_DATA_PATH,
    output_path: str = CNN_MODEL_PATH,
    verbose: int = 1
) -> Dict[str, Any]:
    """
    Train the DNA classification model.
    
    Args:
        epochs: Maximum number of epochs
        batch_size: Batch size
        val_split: Validation split ratio
        seed: Random seed
        data_path: Path to training data
        output_path: Path to save model
        verbose: Verbosity level
        
    Returns:
        Training history dictionary
    """
    # Set random seeds for reproducibility
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
    # Load data
    X_train, X_val, y_train, y_val = load_and_prepare_data(
        data_path, val_split, seed
    )
    
    # Build model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_cnn_model(input_shape)
    
    logger.info("\nModel Architecture:")
    model.summary(print_fn=lambda x: logger.info(x))
    
    # Define callbacks
    callbacks = [
        TrainingLogger(),
        EarlyStopping(
            monitor='val_loss',
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=MIN_LEARNING_RATE,
            verbose=1
        ),
        ModelCheckpoint(
            output_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train model
    logger.info(f"\nStarting training for up to {epochs} epochs...")
    start_time = time.time()
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose
    )
    
    elapsed_time = time.time() - start_time
    logger.info(f"\nTraining completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    
    # Evaluate model
    train_metrics = evaluate_model(model, X_train, y_train, "Training")
    val_metrics = evaluate_model(model, X_val, y_val, "Validation")
    
    # Save history and metrics
    save_training_history(
        history.history,
        {'train': train_metrics, 'validation': val_metrics},
        HISTORY_PATH
    )
    
    logger.info(f"\nModel saved to: {output_path}")
    
    return {
        'history': history.history,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'training_time': elapsed_time
    }


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Validate arguments
    if args.epochs <= 0:
        raise ValueError("Number of epochs must be positive")
    if args.batch_size <= 0:
        raise ValueError("Batch size must be positive")
    if not 0 < args.validation_split < 1:
        raise ValueError("Validation split must be between 0 and 1")
    
    try:
        result = train_model(
            epochs=args.epochs,
            batch_size=args.batch_size,
            val_split=args.validation_split,
            seed=args.seed,
            data_path=args.data,
            output_path=args.output,
            verbose=args.verbose
        )
        print("\n✓ Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()