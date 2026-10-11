import random
import pandas as pd

DNA_ALPHABET = ['A', 'T', 'G', 'C']

def random_dna_sequence(length=100):
    return ''.join(random.choices(DNA_ALPHABET, k=length))

def insert_motif(sequence, motif="TATAAA", position=None):
    if position is None:
        position = random.randint(0, len(sequence) - len(motif))
    return sequence[:position] + motif + sequence[position+len(motif):]

def generate_synthetic_dna_data(num_samples=1000, seq_length=100):
    sequences = []
    labels = []

    for _ in range(num_samples):
        if random.random() > 0.5:
            seq = random_dna_sequence(seq_length)
            seq = insert_motif(seq)
            labels.append(1)
        else:
            seq = random_dna_sequence(seq_length)
            labels.append(0)

        sequences.append(seq)

    df = pd.DataFrame({'sequence': sequences, 'label': labels})
    df.to_csv("synthetic_data.csv", index=False)
    print("Synthetic data saved to synthetic_data.csv")


if __name__ == "__main__":
    generate_synthetic_dna_data()