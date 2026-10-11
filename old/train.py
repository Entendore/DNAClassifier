import os
import time
import json
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from utils import one_hot_encode_sequences

# === Training Configuration ===
DATA_PATH = "synthetic_data.csv"
MODEL_PATH = "dna_cnn_model.keras"
HISTORY_PATH = "history.json"
EPOCHS = 1000
BATCH_SIZE = 32
VAL_SPLIT = 0.2
RANDOM_SEED = 42
VERBOSE = 1

def build_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(64, 8, activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(128, 8, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"'{DATA_PATH}' not found. Please generate synthetic data first.")

    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    sequences = df['sequence'].values
    labels = df['label'].values

    X = one_hot_encode_sequences(sequences)
    y = labels
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=VAL_SPLIT, random_state=RANDOM_SEED)

    print(f"Data loaded. Training on {len(X_train)} samples, validating on {len(X_val)} samples.")
    model = build_model(input_shape=(X.shape[1], 4))

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True)
    ]

    print("Starting training...")
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=VERBOSE
    )
    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.2f} seconds.")

    print(f"Best model saved as '{MODEL_PATH}'")
    
    # Save training history
    history_dict = history.history
    if history_dict:
      with open(HISTORY_PATH, "w") as f:
        json.dump(history_dict, f)
      print(f"Training history saved to '{HISTORY_PATH}'")
    else:
      print("Warning: No history to save.")

    # Print final metrics
    final_train_acc = history_dict['accuracy'][-1]
    final_val_acc = history_dict['val_accuracy'][-1]
    print(f"Final Training Accuracy: {final_train_acc:.4f}")
    print(f"Final Validation Accuracy: {final_val_acc:.4f}")

    

    return history_dict

if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        print(f"Error during training: {e}")
