"""
DNA Sequence Classifier Dashboard

A professional Streamlit application for DNA sequence classification,
data generation, model training, and analysis.

Author: Your Name
Version: 2.0.0
"""

import streamlit as st
import subprocess
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime

# === Initialize Configuration ===
from config import (
    initialize_directories, APP_TITLE, APP_ICON, APP_LAYOUT, PAGE_TITLE,
    SYNTHETIC_DATA_PATH, W2V_MODEL_PATH, CNN_MODEL_PATH, HISTORY_PATH,
    DEFAULT_NUM_SAMPLES, DEFAULT_SEQ_LENGTH, MIN_SEQ_LENGTH, MAX_SEQ_LENGTH,
    CLASSIFICATION_THRESHOLD, COLORS, DNA_ALPHABET
)
from utils import (
    is_valid_dna, sanitize_sequence, calculate_gc_content,
    calculate_nucleotide_frequencies, sequence_to_one_hot,
    load_models, is_model_ready, build_cnn_model, analyze_dataset,
    one_hot_encode_sequences, get_kmers
)

# Initialize directories on import
initialize_directories()

# === Page Configuration ===
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=APP_ICON,
    layout=APP_LAYOUT,
    initial_sidebar_state="expanded"
)

# === Custom CSS ===
def load_custom_css():
    """Apply custom CSS styling."""
    st.markdown("""
    <style>
        /* Main container */
        .main-container {
            padding: 2rem;
        }
        
        /* Card styling */
        .card {
            background-color: white;
            border-radius: 10px;
            padding: 1.5rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        }
        
        /* Header styling */
        .header-title {
            font-size: 2.5rem;
            font-weight: 700;
            color: #1a1a2e;
            margin-bottom: 0.5rem;
        }
        
        .header-subtitle {
            font-size: 1.1rem;
            color: #666;
            margin-bottom: 2rem;
        }
        
        /* Metric cards */
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            padding: 1.5rem;
            text-align: center;
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: 700;
        }
        
        .metric-label {
            font-size: 0.9rem;
            opacity: 0.9;
        }
        
        /* Status indicators */
        .status-ready {
            color: #43A047;
            font-weight: 600;
        }
        
        .status-not-ready {
            color: #E53935;
            font-weight: 600;
        }
        
        /* Code block styling */
        .code-block {
            background-color: #f5f5f5;
            border-radius: 5px;
            padding: 1rem;
            font-family: 'Courier New', monospace;
            font-size: 0.85rem;
            overflow-x: auto;
        }
        
        /* Sequence display */
        .sequence-display {
            font-family: 'Courier New', monospace;
            font-size: 0.9rem;
            word-break: break-all;
            line-height: 1.5;
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 5px;
            border-left: 4px solid #667eea;
        }
        
        /* Probability bar */
        .probability-bar {
            height: 30px;
            border-radius: 15px;
            overflow: hidden;
            background-color: #e0e0e0;
        }
        
        .probability-fill {
            height: 100%;
            border-radius: 15px;
            transition: width 0.5s ease;
        }
        
        /* Hide default Streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Sidebar styling */
        .sidebar-section {
            margin-bottom: 1.5rem;
            padding-bottom: 1.5rem;
            border-bottom: 1px solid #eee;
        }
        
        /* Button styling */
        .stButton > button {
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
    </style>
    """, unsafe_allow_html=True)

load_custom_css()

# === Session State Initialization ===
@st.cache_resource
def initialize_session_state():
    """Initialize session state variables."""
    return {
        'models_loaded': False,
        'w2v_model': None,
        'cnn_model': None,
        'last_classification': None
    }

session = initialize_session_state()

# === Helper Functions ===
def load_cached_models():
    """Load and cache models."""
    if not session['models_loaded'] and is_model_ready():
        try:
            session['w2v_model'], session['cnn_model'] = load_models()
            session['models_loaded'] = True
        except Exception as e:
            st.error(f"Error loading models: {e}")
            session['models_loaded'] = False
    return session['models_loaded']

def render_sidebar():
    """Render the sidebar navigation and status."""
    
    # App title in sidebar
    st.sidebar.markdown(f"""
    <div style="text-align: center; margin-bottom: 1.5rem;">
        <span style="font-size: 2rem;">{APP_ICON}</span>
        <h3 style="margin: 0; color: #1a1a2e;">{APP_TITLE}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # System status
    st.sidebar.markdown("### System Status")
    
    model_ready = is_model_ready()
    data_ready = os.path.exists(SYNTHETIC_DATA_PATH)
    
    status_col1, status_col2 = st.sidebar.columns(2)
    
    with status_col1:
        st.markdown("**Models**")
        if model_ready:
            st.markdown('<span class="status-ready">✓ Ready</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-not-ready">✗ Not Ready</span>', unsafe_allow_html=True)
    
    with status_col2:
        st.markdown("**Data**")
        if data_ready:
            st.markdown('<span class="status-ready">✓ Ready</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-not-ready">✗ Not Ready</span>', unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    # Navigation
    st.sidebar.markdown("### Navigation")
    
    page_options = [
        ("classify", "🔍 Classify Sequence", "Classify DNA sequences"),
        ("batch_classify", "📋 Batch Classification", "Process multiple sequences"),
        ("generate", "🧪 Generate Data", "Create synthetic data"),
        ("train_w2v", "🧠 Train Word2Vec", "Train embeddings"),
        ("train_cnn", "⚙️ Train CNN Model", "Train classifier"),
        ("history", "📊 Training History", "View training metrics"),
        ("data_view", "📂 Data Explorer", "View and analyze data"),
        ("settings", "🔧 Settings", "Configuration options")
    ]
    
    selected_page = None
    for page_id, page_name, page_desc in page_options:
        if st.sidebar.button(page_name, key=f"nav_{page_id}", use_container_width=True):
            selected_page = page_id
            st.session_state.current_page = page_id
    
    # Use session state for page persistence
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "classify"
    
    if selected_page:
        st.session_state.current_page = selected_page
    
    st.sidebar.markdown("---")
    
    # Quick info
    st.sidebar.markdown("### Quick Info")
    if data_ready:
        df = pd.read_csv(SYNTHETIC_DATA_PATH)
        st.sidebar.info(f"📊 {len(df)} samples loaded")
    if model_ready:
        st.sidebar.info(f"🧠 Models available")
    
    return st.session_state.current_page

def format_sequence_display(sequence: str, highlight_motif: bool = True) -> str:
    """Format DNA sequence for display with optional motif highlighting."""
    if not highlight_motif:
        return sequence
    
    # Highlight TATA motifs
    motifs = ["TATAAA", "TATATA", "TATAA", "CATAAA", "TATATAA", "TATA"]
    highlighted = sequence
    for motif in motifs:
        highlighted = highlighted.replace(
            motif, 
            f'<span style="background-color: #ffeb3b; padding: 2px; border-radius: 3px;">{motif}</span>'
        )
    return highlighted

def render_classification_result(prediction: float, sequence: str):
    """Render classification results with visualizations."""
    
    label = "Positive (TATA Box Present)" if prediction >= CLASSIFICATION_THRESHOLD else "Negative (No TATA Box)"
    confidence = max(prediction, 1 - prediction)
    
    # Result header
    if prediction >= CLASSIFICATION_THRESHOLD:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #43A047 0%, #66BB6A 100%); 
                    padding: 1.5rem; border-radius: 10px; color: white; margin-bottom: 1rem;">
            <h2 style="margin: 0;">✓ POSITIVE</h2>
            <p style="margin: 0; opacity: 0.9;">TATA Box Motif Detected</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1E88E5 0%, #42A5F5 100%); 
                    padding: 1.5rem; border-radius: 10px; color: white; margin-bottom: 1rem;">
            <h2 style="margin: 0;">✗ NEGATIVE</h2>
            <p style="margin: 0; opacity: 0.9;">No TATA Box Motif Detected</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Metrics row
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Prediction Score", f"{prediction:.4f}")
    
    with col2:
        st.metric("Confidence", f"{confidence:.2%}")
    
    with col3:
        st.metric("Threshold", f"{CLASSIFICATION_THRESHOLD:.2f}")
    
    # Probability visualization
    st.markdown("### Probability Distribution")
    
    fig, ax = plt.subplots(figsize=(10, 2))
    
    # Background bar
    ax.barh(0, 1, height=0.5, color='#e0e0e0', edgecolor='none')
    
    # Positive probability
    pos_color = '#43A047' if prediction >= CLASSIFICATION_THRESHOLD else '#81C784'
    ax.barh(0, prediction, height=0.5, color=pos_color, edgecolor='none', label='Positive')
    
    # Threshold marker
    ax.axvline(x=CLASSIFICATION_THRESHOLD, color='#E53935', linestyle='--', linewidth=2, label='Threshold')
    
    # Labels
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel('Probability')
    ax.set_yticks([])
    ax.legend(loc='upper right')
    
    # Add text labels
    ax.text(prediction/2, 0, f'{prediction:.2%}', ha='center', va='center', 
            fontweight='bold', color='white', fontsize=12)
    ax.text((1+prediction)/2, 0, f'{(1-prediction):.2%}', ha='center', va='center', 
            fontweight='bold', color='#666', fontsize=12)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Sequence analysis
    st.markdown("### Sequence Analysis")
    
    gc_content = calculate_gc_content(sequence)
    nucleotide_freqs = calculate_nucleotide_frequencies(sequence)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Sequence Display**")
        st.markdown(
            f'<div class="sequence-display">{format_sequence_display(sequence)}</div>',
            unsafe_allow_html=True
        )
        st.markdown(f"**Length:** {len(sequence)} bp")
    
    with col2:
        st.markdown("**Nucleotide Composition**")
        
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ['#E53935', '#1E88E5', '#43A047', '#FB8C00']  # A, C, G, T
        bars = ax.bar(
            ['A', 'C', 'G', 'T'],
            [nucleotide_freqs[b] for b in DNA_ALPHABET],
            color=colors
        )
        
        # Add value labels
        for bar, val in zip(bars, [nucleotide_freqs[b] for b in DNA_ALPHABET]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.1%}', ha='center', va='bottom', fontsize=10)
        
        ax.set_ylabel('Frequency')
        ax.set_ylim(0, 0.5)
        ax.set_title(f'GC Content: {gc_content:.1%}', fontsize=12)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ============================================================================
# Page Renderers
# ============================================================================

def render_classify_page():
    """Render the single sequence classification page."""
    st.markdown("""
    <div class="header-title">🔍 Classify DNA Sequence</div>
    <div class="header-subtitle">Analyze individual DNA sequences for TATA box promoter motifs</div>
    """, unsafe_allow_html=True)
    
    if not is_model_ready():
        st.warning("⚠️ Models not ready. Please train the models first.")
        st.info("Go to **🧠 Train Word2Vec** then **⚙️ Train CNN Model** to prepare the system.")
        return
    
    # Input section
    col1, col2 = st.columns([3, 1])
    
    with col1:
        sequence_input = st.text_area(
            "Enter DNA Sequence",
            placeholder="Example: ATCGATCGATCGTATAAAATCGATCGATCG...",
            height=150,
            max_chars=1000
        )
    
    with col2:
        st.markdown("**Quick Actions**")
        
        if st.button("📝 Example Sequence", use_container_width=True):
            example = "ATCGATCGATCGTATAAAATCGATCGATCGATCGATCGATCGTATAAAATCGATCGATCG"
            st.session_state.example_seq = example
            st.rerun()
        
        if st.button("🎲 Random Sequence", use_container_width=True):
            import random
            random_seq = ''.join(random.choices(DNA_ALPHABET, k=100))
            st.session_state.example_seq = random_seq
            st.rerun()
        
        st.markdown("---")
        st.markdown("**Validation**")
        if sequence_input:
            if is_valid_dna(sequence_input):
                st.success("✓ Valid DNA sequence")
                st.markdown(f"Length: {len(sequence_input)} bp")
            else:
                st.error("✗ Invalid characters detected")
    
    # Use example if set
    if 'example_seq' in st.session_state and not sequence_input:
        sequence_input = st.session_state.example_seq
    
    # Classification button
    if st.button("🧬 Classify Sequence", type="primary", use_container_width=True):
        if not sequence_input:
            st.error("Please enter a DNA sequence.")
            return
        
        if not is_valid_dna(sequence_input):
            st.error("Invalid DNA sequence. Use only A, T, G, C characters.")
            return
        
        with st.spinner("Analyzing sequence..."):
            try:
                # Load models if needed
                if not load_cached_models():
                    return
                
                # Prepare input
                sequence = sanitize_sequence(sequence_input)
                encoded = sequence_to_one_hot(sequence)
                encoded = encoded.reshape(1, *encoded.shape)
                
                # Predict
                prediction = session['cnn_model'].predict(encoded, verbose=0)[0][0]
                
                # Display results
                render_classification_result(prediction, sequence)
                
                # Store result
                session['last_classification'] = {
                    'sequence': sequence,
                    'prediction': float(prediction),
                    'timestamp': datetime.now().isoformat()
                }
                
            except Exception as e:
                st.error(f"Classification error: {e}")


def render_batch_classify_page():
    """Render the batch classification page."""
    st.markdown("""
    <div class="header-title">📋 Batch Classification</div>
    <div class="header-subtitle">Classify multiple DNA sequences at once</div>
    """, unsafe_allow_html=True)
    
    if not is_model_ready():
        st.warning("⚠️ Models not ready. Please train the models first.")
        return
    
    # Input options
    tab1, tab2 = st.tabs(["Text Input", "File Upload"])
    
    sequences = []
    
    with tab1:
        batch_input = st.text_area(
            "Enter sequences (one per line)",
            placeholder="ATCGATCGTATAAA...\nGCTAGCTAGCTAGC...\n...",
            height=200
        )
        if batch_input:
            sequences = [s.strip() for s in batch_input.split('\n') if s.strip()]
    
    with tab2:
        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=['csv'],
            help="CSV should have a 'sequence' column"
        )
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                if 'sequence' in df.columns:
                    sequences = df['sequence'].dropna().astype(str).tolist()
                else:
                    st.error("CSV must have a 'sequence' column")
            except Exception as e:
                st.error(f"Error reading file: {e}")
    
    if sequences:
        st.info(f"**{len(sequences)} sequences** loaded for classification")
        
        # Validate sequences
        valid_sequences = []
        invalid_count = 0
        
        for seq in sequences:
            if is_valid_dna(seq):
                valid_sequences.append(sanitize_sequence(seq))
            else:
                invalid_count += 1
        
        if invalid_count > 0:
            st.warning(f"⚠️ {invalid_count} invalid sequences will be skipped")
        
        if valid_sequences:
            if st.button("🚀 Classify All Sequences", type="primary"):
                with st.spinner("Processing sequences..."):
                    try:
                        if not load_cached_models():
                            return
                        
                        results = []
                        for i, seq in enumerate(valid_sequences):
                            encoded = sequence_to_one_hot(seq)
                            encoded = encoded.reshape(1, *encoded.shape)
                            prediction = session['cnn_model'].predict(encoded, verbose=0)[0][0]
                            
                            results.append({
                                'index': i + 1,
                                'sequence': seq[:50] + '...' if len(seq) > 50 else seq,
                                'full_sequence': seq,
                                'prediction_score': prediction,
                                'prediction_label': 'Positive' if prediction >= CLASSIFICATION_THRESHOLD else 'Negative',
                                'gc_content': calculate_gc_content(seq),
                                'length': len(seq)
                            })
                        
                        results_df = pd.DataFrame(results)
                        
                        # Display results
                        st.markdown("### Classification Results")
                        
                        # Summary metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            positive_count = sum(1 for r in results if r['prediction_label'] == 'Positive')
                            st.metric("Positive", f"{positive_count}", f"{positive_count/len(results):.1%}")
                        with col2:
                            negative_count = len(results) - positive_count
                            st.metric("Negative", f"{negative_count}", f"{negative_count/len(results):.1%}")
                        with col3:
                            avg_score = np.mean([r['prediction_score'] for r in results])
                            st.metric("Avg Score", f"{avg_score:.4f}")
                        
                        # Results table
                        st.dataframe(
                            results_df[['index', 'sequence', 'prediction_score', 'prediction_label', 'gc_content', 'length']],
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        # Distribution plot
                        fig, ax = plt.subplots(figsize=(10, 4))
                        ax.hist([r['prediction_score'] for r in results], bins=20, edgecolor='black', alpha=0.7)
                        ax.axvline(x=CLASSIFICATION_THRESHOLD, color='red', linestyle='--', label='Threshold')
                        ax.set_xlabel('Prediction Score')
                        ax.set_ylabel('Count')
                        ax.set_title('Distribution of Prediction Scores')
                        ax.legend()
                        st.pyplot(fig)
                        plt.close()
                        
                        # Download button
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Results",
                            csv,
                            "classification_results.csv",
                            "text/csv"
                        )
                        
                    except Exception as e:
                        st.error(f"Batch classification error: {e}")


def render_generate_data_page():
    """Render the synthetic data generation page."""
    st.markdown("""
    <div class="header-title">🧪 Generate Synthetic Data</div>
    <div class="header-subtitle">Create synthetic DNA sequences with and without TATA box motifs</div>
    """, unsafe_allow_html=True)
    
    # Parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        num_samples = st.number_input(
            "Number of Samples",
            min_value=100,
            max_value=100000,
            value=DEFAULT_NUM_SAMPLES,
            step=100
        )
    
    with col2:
        seq_length = st.number_input(
            "Sequence Length",
            min_value=MIN_SEQ_LENGTH,
            max_value=MAX_SEQ_LENGTH,
            value=DEFAULT_SEQ_LENGTH,
            step=10
        )
    
    with col3:
        noise_prob = st.slider(
            "Mutation Noise",
            min_value=0.0,
            max_value=0.1,
            value=0.02,
            step=0.005,
            help="Probability of random base mutations"
        )
    
    # Information
    st.markdown("""
    ### About Synthetic Data Generation
    
    The generator creates DNA sequences with the following characteristics:
    - **Positive samples**: Contains TATA box promoter motifs (TATAAA, TATATA, etc.)
    - **Negative samples**: Random sequences without promoter motifs
    - **Noise**: Optional random mutations for realism
    """)
    
    if st.button("🧬 Generate Dataset", type="primary", use_container_width=True):
        with st.spinner("Generating synthetic data..."):
            try:
                result = subprocess.run(
                    [
                        sys.executable, "generatesynthdata.py",
                        "--samples", str(num_samples),
                        "--length", str(seq_length),
                        "--noise", str(noise_prob),
                        "--output", SYNTHETIC_DATA_PATH
                    ],
                    capture_output=True,
                    text=True,
                    encoding="utf-8"
                )
                
                if result.returncode == 0:
                    st.success("✓ Synthetic data generated successfully!")
                    st.code(result.stdout)
                    
                    # Show preview
                    df = pd.read_csv(SYNTHETIC_DATA_PATH)
                    st.markdown("### Data Preview")
                    st.dataframe(df.head(5), use_container_width=True)
                    
                    # Quick stats
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Samples", len(df))
                    with col2:
                        st.metric("Positive", sum(df['label'] == 1))
                    with col3:
                        st.metric("Negative", sum(df['label'] == 0))
                    
                    st.rerun()
                else:
                    st.error("Failed to generate data")
                    if result.stderr:
                        st.code(result.stderr)
                        
            except Exception as e:
                st.error(f"Error: {e}")


def render_train_w2v_page():
    """Render the Word2Vec training page."""
    st.markdown("""
    <div class="header-title">🧠 Train Word2Vec Embeddings</div>
    <div class="header-subtitle">Train word embeddings on DNA k-mers</div>
    """, unsafe_allow_html=True)
    
    if not os.path.exists(SYNTHETIC_DATA_PATH):
        st.warning("⚠️ No synthetic data found. Please generate data first.")
        return
    
    # Load data for preview
    df = pd.read_csv(SYNTHETIC_DATA_PATH)
    
    st.markdown("### Training Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        kmer_size = st.selectbox(
            "K-mer Size",
            options=[3, 4, 5, 6],
            index=0,
            help="Size of k-mers for tokenization"
        )
        
        vector_size = st.selectbox(
            "Vector Dimension",
            options=[50, 100, 150, 200],
            index=1,
            help="Dimensionality of embedding vectors"
        )
    
    with col2:
        window_size = st.slider(
            "Context Window",
            min_value=3,
            max_value=15,
            value=5
        )
        
        epochs = st.slider(
            "Training Epochs",
            min_value=10,
            max_value=100,
            value=50
        )
    
    st.markdown("""
    ### About Word2Vec for DNA
    
    Word2Vec learns distributed representations of k-mers by predicting 
    context in DNA sequences. This captures semantic relationships between 
    similar k-mer patterns.
    """)
    
    # Preview tokenization
    st.markdown("### K-mer Tokenization Preview")
    sample_seq = df['sequence'].iloc[0][:30]
    kmers = get_kmers(sample_seq, kmer_size)
    
    st.markdown(f"**Sequence:** `{sample_seq}...`")
    st.markdown(f"**K-mers (k={kmer_size}):** `{', '.join(kmers)}...`")
    
    if st.button("🧠 Train Word2Vec Model", type="primary", use_container_width=True):
        with st.spinner("Training Word2Vec model..."):
            try:
                from utils import train_word2vec
                
                model = train_word2vec(
                    df["sequence"].values,
                    k=kmer_size,
                    vector_size=vector_size,
                    window=window_size,
                    epochs=epochs
                )
                
                st.success("✓ Word2Vec model trained successfully!")
                
                # Model statistics
                st.markdown("### Model Statistics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Vocabulary Size", len(model.wv))
                with col2:
                    st.metric("Vector Size", vector_size)
                with col3:
                    st.metric("K-mer Size", kmer_size)
                
                # Show similar k-mers
                st.markdown("### Similar K-mers Example")
                if 'TAT' in model.wv:
                    similar = model.wv.most_similar('TAT', topn=5)
                    for kmer, score in similar:
                        st.markdown(f"- `{kmer}`: {score:.4f}")
                
                st.rerun()
                
            except Exception as e:
                st.error(f"Training error: {e}")


def render_train_cnn_page():
    """Render the CNN training page."""
    st.markdown("""
    <div class="header-title">⚙️ Train CNN Model</div>
    <div class="header-subtitle">Train convolutional neural network for DNA classification</div>
    """, unsafe_allow_html=True)
    
    # Check prerequisites
    if not os.path.exists(SYNTHETIC_DATA_PATH):
        st.warning("⚠️ No synthetic data found. Please generate data first.")
        return
    
    # Training parameters
    st.markdown("### Training Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        epochs = st.number_input(
            "Max Epochs",
            min_value=10,
            max_value=5000,
            value=1000,
            step=100
        )
    
    with col2:
        batch_size = st.selectbox(
            "Batch Size",
            options=[16, 32, 64, 128],
            index=1
        )
    
    with col3:
        val_split = st.slider(
            "Validation Split",
            min_value=0.1,
            max_value=0.3,
            value=0.2,
            step=0.05
        )
    
    # Advanced options
    with st.expander("Advanced Options"):
        learning_rate = st.number_input(
            "Learning Rate",
            min_value=0.0001,
            max_value=0.01,
            value=0.001,
            format="%.4f"
        )
        
        early_stop_patience = st.number_input(
            "Early Stopping Patience",
            min_value=5,
            max_value=50,
            value=15
        )
    
    # Model architecture preview
    st.markdown("### Model Architecture")
    
    if os.path.exists(SYNTHETIC_DATA_PATH):
        df = pd.read_csv(SYNTHETIC_DATA_PATH)
        seq_len = len(df['sequence'].iloc[0])
        
        model = build_cnn_model((seq_len, 4))
        
        # Display model summary in a nice format
        summary_lines = []
        model.summary(print_fn=lambda x: summary_lines.append(x))
        
        st.code('\n'.join(summary_lines), language='text')
    
    st.markdown("---")
    
    if st.button("🚀 Start Training", type="primary", use_container_width=True):
        with st.spinner("Training model... (this may take a while)"):
            try:
                process = subprocess.run(
                    [
                        sys.executable, "train.py",
                        "--epochs", str(epochs),
                        "--batch_size", str(batch_size),
                        "--validation_split", str(val_split),
                        "--verbose", "1"
                    ],
                    capture_output=True,
                    text=True,
                    encoding="utf-8"
                )
                
                st.markdown("### Training Output")
                
                # Show output in expandable
                with st.expander("View Training Log", expanded=False):
                    st.code(process.stdout)
                
                if process.returncode == 0:
                    st.success("✓ Model trained and saved successfully!")
                    
                    # Load and display results
                    if os.path.exists(HISTORY_PATH):
                        with open(HISTORY_PATH, 'r') as f:
                            history_data = json.load(f)
                        
                        if 'evaluation_metrics' in history_data:
                            metrics = history_data['evaluation_metrics'].get('validation', {})
                            
                            st.markdown("### Validation Metrics")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
                            with col2:
                                st.metric("Precision", f"{metrics.get('precision', 0):.4f}")
                            with col3:
                                st.metric("Recall", f"{metrics.get('recall', 0):.4f}")
                            with col4:
                                st.metric("F1 Score", f"{metrics.get('f1_score', 0):.4f}")
                    
                    st.rerun()
                else:
                    st.error("Training failed!")
                    if process.stderr:
                        with st.expander("Error Details"):
                            st.code(process.stderr)
                        
            except Exception as e:
                st.error(f"Training error: {e}")


def render_training_history_page():
    """Render the training history visualization page."""
    st.markdown("""
    <div class="header-title">📊 Training History</div>
    <div class="header-subtitle">Visualize model training progress and performance</div>
    """, unsafe_allow_html=True)
    
    if not os.path.exists(HISTORY_PATH):
        st.warning("⚠️ No training history found. Please train the model first.")
        return
    
    try:
        with open(HISTORY_PATH, 'r') as f:
            history_data = json.load(f)
        
        history = history_data.get('training_history', {})
        metadata = history_data.get('metadata', {})
        metrics = history_data.get('evaluation_metrics', {})
        
        if not history:
            raise ValueError("Training history is empty")
        
        # Summary metrics
        st.markdown("### Training Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Epochs", metadata.get('total_epochs', 'N/A'))
        with col2:
            st.metric("Best Val Accuracy", f"{metadata.get('best_val_accuracy', 0):.4f}")
        with col3:
            st.metric("Best Val Loss", f"{metadata.get('best_val_loss', 0):.4f}")
        with col4:
            if 'timestamp' in metadata:
                st.metric("Last Trained", metadata['timestamp'][:10])
        
        # Plot accuracy
        st.markdown("### Accuracy Curves")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy plot
        axes[0].plot(history.get('accuracy', []), label='Training', color='#1E88E5')
        axes[0].plot(history.get('val_accuracy', []), label='Validation', color='#E53935')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Loss plot
        axes[1].plot(history.get('loss', []), label='Training', color='#1E88E5')
        axes[1].plot(history.get('val_loss', []), label='Validation', color='#E53935')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Model Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # AUC plot if available
        if 'auc' in history or 'val_auc' in history:
            st.markdown("### AUC Curves")
            fig, ax = plt.subplots(figsize=(10, 5))
            if 'auc' in history:
                ax.plot(history['auc'], label='Training AUC', color='#1E88E5')
            if 'val_auc' in history:
                ax.plot(history['val_auc'], label='Validation AUC', color='#E53935')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('AUC')
            ax.set_title('Area Under Curve (AUC)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
        
        # Detailed metrics
        if metrics:
            st.markdown("### Detailed Evaluation Metrics")
            
            for dataset_name, dataset_metrics in metrics.items():
                with st.expander(f"{dataset_name.capitalize()} Set Metrics"):
                    cols = st.columns(len(dataset_metrics))
                    for i, (metric_name, metric_value) in enumerate(dataset_metrics.items()):
                        with cols[i]:
                            st.metric(
                                metric_name.replace('_', ' ').title(),
                                f"{metric_value:.4f}"
                            )
        
        # Learning rate plot
        if 'lr' in history:
            st.markdown("### Learning Rate Schedule")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(history['lr'], color='#667eea')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning Rate')
            ax.set_title('Learning Rate Over Training')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
        
    except json.JSONDecodeError:
        st.error("Training history file is corrupted. Please retrain the model.")
    except Exception as e:
        st.error(f"Error loading training history: {e}")


def render_data_explorer_page():
    """Render the data exploration page."""
    st.markdown("""
    <div class="header-title">📂 Data Explorer</div>
    <div class="header-subtitle">View, analyze, and export synthetic DNA data</div>
    """, unsafe_allow_html=True)
    
    if not os.path.exists(SYNTHETIC_DATA_PATH):
        st.warning("⚠️ No synthetic data found. Please generate data first.")
        return
    
    df = pd.read_csv(SYNTHETIC_DATA_PATH)
    
    # Dataset statistics
    stats = analyze_dataset(df)
    
    st.markdown("### Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Samples", stats['total_samples'])
    with col2:
        st.metric("Positive", stats['label_distribution']['positive'])
    with col3:
        st.metric("Negative", stats['label_distribution']['negative'])
    with col4:
        st.metric("TATA Box Rate", f"{stats['tata_box_ratio']:.1%}")
    
    # Distribution plots
    st.markdown("### Data Distribution")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Label distribution
    label_counts = df['label'].value_counts()
    axes[0, 0].pie(
        label_counts.values,
        labels=['Negative', 'Positive'],
        autopct='%1.1f%%',
        colors=['#1E88E5', '#E53935'],
        startangle=90
    )
    axes[0, 0].set_title('Label Distribution')
    
    # Sequence length distribution
    lengths = df['sequence'].apply(len)
    axes[0, 1].hist(lengths, bins=30, edgecolor='black', alpha=0.7, color='#667eea')
    axes[0, 1].set_xlabel('Sequence Length')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Sequence Length Distribution')
    
    # GC content distribution
    gc_contents = df['sequence'].apply(calculate_gc_content)
    axes[1, 0].hist(gc_contents, bins=30, edgecolor='black', alpha=0.7, color='#43A047')
    axes[1, 0].set_xlabel('GC Content')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('GC Content Distribution')
    
    # Nucleotide frequency
    all_freqs = df['sequence'].apply(calculate_nucleotide_frequencies)
    freq_df = pd.DataFrame(list(all_freqs))
    freq_df.mean().plot(kind='bar', ax=axes[1, 1], color=['#E53935', '#1E88E5', '#43A047', '#FB8C00'])
    axes[1, 1].set_xlabel('Nucleotide')
    axes[1, 1].set_ylabel('Average Frequency')
    axes[1, 1].set_title('Average Nucleotide Frequency')
    axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=0)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Sample data viewer
    st.markdown("### Sample Data")
    
    num_samples = st.slider("Samples to display", min_value=5, max_value=50, value=10)
    sample_df = df.sample(min(num_samples, len(df)), random_state=42)
    
    # Format sequence column for display
    display_df = sample_df.copy()
    display_df['sequence'] = display_df['sequence'].str[:50] + '...'
    display_df['label'] = display_df['label'].map({0: 'Negative', 1: 'Positive'})
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Download options
    st.markdown("### Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            "📥 Download Full Dataset (CSV)",
            df.to_csv(index=False),
            "synthetic_data.csv",
            "text/csv",
            use_container_width=True
        )
    
    with col2:
        # JSON export
        json_data = df.to_dict(orient='records')
        st.download_button(
            "📥 Download as JSON",
            json.dumps(json_data, indent=2),
            "synthetic_data.json",
            "application/json",
            use_container_width=True
        )


def render_settings_page():
    """Render the settings page."""
    st.markdown("""
    <div class="header-title">🔧 Settings</div>
    <div class="header-subtitle">System configuration and maintenance</div>
    """, unsafe_allow_html=True)
    
    st.markdown("### System Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**File Locations**")
        st.code(f"""
Data:      {SYNTHETIC_DATA_PATH}
Word2Vec:  {W2V_MODEL_PATH}
CNN Model: {CNN_MODEL_PATH}
History:   {HISTORY_PATH}
        """, language='text')
    
    with col2:
        st.markdown("**File Status**")
        
        files = [
            ("Synthetic Data", SYNTHETIC_DATA_PATH),
            ("Word2Vec Model", W2V_MODEL_PATH),
            ("CNN Model", CNN_MODEL_PATH),
            ("Training History", HISTORY_PATH)
        ]
        
        for name, path in files:
            exists = os.path.exists(path)
            status = "✓ Exists" if exists else "✗ Missing"
            if exists:
                size = os.path.getsize(path)
                size_str = f"({size / 1024:.1f} KB)" if size < 1024*1024 else f"({size / (1024*1024):.1f} MB)"
                st.markdown(f"- **{name}**: {status} {size_str}")
            else:
                st.markdown(f"- **{name}**: {status}")
    
    st.markdown("---")
    
    st.markdown("### Maintenance Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear Training History", use_container_width=True):
            if os.path.exists(HISTORY_PATH):
                os.remove(HISTORY_PATH)
                st.success("Training history cleared.")
                st.rerun()
            else:
                st.info("No training history to clear.")
    
    with col2:
        if st.button("🗑️ Clear All Data & Models", type="primary", use_container_width=True):
            files_to_remove = [SYNTHETIC_DATA_PATH, W2V_MODEL_PATH, CNN_MODEL_PATH, HISTORY_PATH]
            removed = 0
            for f in files_to_remove:
                if os.path.exists(f):
                    os.remove(f)
                    removed += 1
            st.warning(f"Removed {removed} files. Please regenerate data and retrain models.")
            st.rerun()
    
    st.markdown("---")
    
    st.markdown("### Classification Threshold")
    
    current_threshold = CLASSIFICATION_THRESHOLD
    st.info(f"Current threshold: **{current_threshold}** (defined in config.py)")
    st.markdown("""
    Sequences with prediction scores above this threshold are classified as 
    positive (containing TATA box motif). Adjust in `config.py` to modify.
    """)


# ============================================================================
# Main Application
# ============================================================================

def main():
    """Main application entry point."""
    
    # Render sidebar
    current_page = render_sidebar()
    
    # Main content area
    main_container = st.container()
    
    with main_container:
        # Route to appropriate page
        page_renderers = {
            'classify': render_classify_page,
            'batch_classify': render_batch_classify_page,
            'generate': render_generate_data_page,
            'train_w2v': render_train_w2v_page,
            'train_cnn': render_train_cnn_page,
            'history': render_training_history_page,
            'data_view': render_data_explorer_page,
            'settings': render_settings_page
        }
        
        renderer = page_renderers.get(current_page, render_classify_page)
        renderer()


if __name__ == "__main__":
    main()