"""
Synthetic DNA Data Generator

Generates realistic synthetic DNA sequences with and without TATA box motifs
for training classification models.

Usage:
    python generatesynthdata.py --samples 1000 --length 100 --output data/synthetic_data.csv
"""

import argparse
import random
import pandas as pd
import numpy as np
from typing import List, Tuple
from datetime import datetime

from config import (
    DNA_ALPHABET, TATA_MOTIFS, NOISE_PROBABILITY,
    DEFAULT_NUM_SAMPLES, DEFAULT_SEQ_LENGTH, SYNTHETIC_DATA_PATH
)


class DNASynthesizer:
    """Class for generating synthetic DNA sequences."""
    
    def __init__(
        self,
        num_samples: int = DEFAULT_NUM_SAMPLES,
        seq_length: int = DEFAULT_SEQ_LENGTH,
        noise_probability: float = NOISE_PROBABILITY,
        motifs: List[str] = None
    ):
        """
        Initialize the DNA synthesizer.
        
        Args:
            num_samples: Number of samples to generate
            seq_length: Length of each sequence
            noise_probability: Probability of random base mutation
            motifs: List of promoter motifs to insert
        """
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.noise_probability = noise_probability
        self.motifs = motifs or TATA_MOTIFS
    
    def generate_random_sequence(self, length: int = None) -> str:
        """
        Generate a random DNA sequence.
        
        Args:
            length: Length of the sequence (uses default if None)
            
        Returns:
            Random DNA sequence string
        """
        length = length or self.seq_length
        return ''.join(random.choices(DNA_ALPHABET, k=length))
    
    def introduce_noise(self, sequence: str) -> str:
        """
        Introduce random mutations to a sequence.
        
        Args:
            sequence: DNA sequence
            
        Returns:
            Sequence with potential mutations
        """
        seq_list = list(sequence)
        for i in range(len(seq_list)):
            if random.random() < self.noise_probability:
                # Replace with a different base
                original = seq_list[i]
                alternatives = [b for b in DNA_ALPHABET if b != original]
                seq_list[i] = random.choice(alternatives)
        return ''.join(seq_list)
    
    def insert_motif(
        self,
        sequence: str,
        motif: str = None,
        position: int = None
    ) -> str:
        """
        Insert a motif into a sequence at a specified or random position.
        
        Args:
            sequence: Original DNA sequence
            motif: Motif to insert (uses random if None)
            position: Position to insert at (uses random if None)
            
        Returns:
            Sequence with inserted motif
        """
        motif = motif or random.choice(self.motifs)
        
        # Ensure sequence is long enough
        if len(sequence) < len(motif):
            return sequence
        
        max_position = len(sequence) - len(motif)
        if position is None:
            position = random.randint(0, max_position)
        
        return sequence[:position] + motif + sequence[position + len(motif):]
    
    def generate_positive_sample(self) -> str:
        """
        Generate a positive sample (contains TATA box motif).
        
        Returns:
            DNA sequence with inserted motif
        """
        sequence = self.generate_random_sequence()
        sequence = self.insert_motif(sequence)
        sequence = self.introduce_noise(sequence)
        return sequence
    
    def generate_negative_sample(self) -> str:
        """
        Generate a negative sample (no TATA box motif).
        
        Returns:
            Random DNA sequence without motif
        """
        sequence = self.generate_random_sequence()
        
        # Ensure no TATA variants are present
        while any(motif in sequence for motif in self.motifs):
            sequence = self.generate_random_sequence()
        
        sequence = self.introduce_noise(sequence)
        return sequence
    
    def generate_dataset(self) -> pd.DataFrame:
        """
        Generate the complete dataset.
        
        Returns:
            DataFrame with sequences and labels
        """
        sequences = []
        labels = []
        
        for _ in range(self.num_samples):
            if random.random() > 0.5:
                sequences.append(self.generate_positive_sample())
                labels.append(1)
            else:
                sequences.append(self.generate_negative_sample())
                labels.append(0)
        
        df = pd.DataFrame({
            'sequence': sequences,
            'label': labels
        })
        
        # Shuffle the dataset
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        return df
    
    def save_dataset(self, df: pd.DataFrame, output_path: str) -> None:
        """
        Save dataset to CSV file.
        
        Args:
            df: DataFrame to save
            output_path: Path to save the file
        """
        df.to_csv(output_path, index=False)
        print(f"✓ Synthetic data saved to: {output_path}")
        print(f"  - Total samples: {len(df)}")
        print(f"  - Positive samples: {sum(df['label'] == 1)}")
        print(f"  - Negative samples: {sum(df['label'] == 0)}")
        print(f"  - Sequence length: {len(df['sequence'].iloc[0])}")
        print(f"  - Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate synthetic DNA sequences for classification training.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '-n', '--samples',
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help='Number of samples to generate'
    )
    
    parser.add_argument(
        '-l', '--length',
        type=int,
        default=DEFAULT_SEQ_LENGTH,
        help='Length of each DNA sequence'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=SYNTHETIC_DATA_PATH,
        help='Output file path'
    )
    
    parser.add_argument(
        '--noise',
        type=float,
        default=NOISE_PROBABILITY,
        help='Probability of random base mutation (0-1)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Validate arguments
    if args.samples <= 0:
        raise ValueError("Number of samples must be positive")
    if args.length < 10:
        raise ValueError("Sequence length must be at least 10")
    if not 0 <= args.noise <= 1:
        raise ValueError("Noise probability must be between 0 and 1")
    
    print("=" * 50)
    print("DNA Synthetic Data Generator")
    print("=" * 50)
    print(f"Generating {args.samples} sequences of length {args.length}...")
    
    synthesizer = DNASynthesizer(
        num_samples=args.samples,
        seq_length=args.length,
        noise_probability=args.noise
    )
    
    df = synthesizer.generate_dataset()
    synthesizer.save_dataset(df, args.output)


if __name__ == "__main__":
    main()