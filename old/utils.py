import numpy as np
import os
from gensim.models import Word2Vec
from tensorflow.keras.models import load_model

# === Constants ===
DNA_BASES = {'A', 'C', 'G', 'T'}
DEFAULT_K = 3
DEFAULT_VECTOR_SIZE = 100
W2V_PATH = "w2v.model"
MODEL_PATH = "dna_cnn_model.keras"

# === One-Hot Encoder ===
def one_hot_encode_sequences(sequences):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seq_len = len(sequences[0])

    one_hot = np.zeros((len(sequences), seq_len, 4), dtype=np.float32)
    for i, seq in enumerate(sequences):
        for j, base in enumerate(seq):
            if base in mapping:
                one_hot[i, j, mapping[base]] = 1
    return one_hot

# === K-mer Tokenization ===
def get_kmers(sequence, k=DEFAULT_K):
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

# === Train Word2Vec ===
def train_word2vec(sequences, k=DEFAULT_K, size=DEFAULT_VECTOR_SIZE):
    tokenized = [get_kmers(seq.upper(), k) for seq in sequences]
    model = Word2Vec(sentences=tokenized, vector_size=size, window=5, min_count=1, workers=4)
    model.save(W2V_PATH)
    print(f"Word2Vec model saved to {W2V_PATH}")
    return model

# === Convert Sequence to Vector Using Word2Vec ===
def sequence_to_vector(sequence, w2v_model, k=DEFAULT_K):
    sequence = sequence.upper()
    kmers = get_kmers(sequence, k)
    vectors = [w2v_model.wv[kmer] for kmer in kmers if kmer in w2v_model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(w2v_model.vector_size)

# === Load Trained Models ===
def load_models(w2v_path=W2V_PATH, model_path=MODEL_PATH):
    if not os.path.exists(w2v_path):
        raise FileNotFoundError("Word2Vec model not found.")
    if not os.path.exists(model_path):
        raise FileNotFoundError("CNN model not found.")
    w2v = Word2Vec.load(w2v_path)
    model = load_model(model_path)
    return w2v, model

# === Validate DNA Sequence ===
def is_valid_dna(sequence):
    return set(sequence.upper()).issubset(DNA_BASES)

# === Check If Models Are Ready ===
def is_model_ready():
    return os.path.exists(W2V_PATH) and os.path.exists(MODEL_PATH)

# === Generate Dummy Vector for Testing ===
def dummy_vector(vector_size=DEFAULT_VECTOR_SIZE):
    return np.random.rand(vector_size)
