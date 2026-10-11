import streamlit as st
import subprocess
import os
import pandas as pd
import matplotlib.pyplot as plt
import json

from utils import load_models, sequence_to_vector

# === Streamlit Config ===
st.set_page_config(page_title="DNA Classifier", layout="wide")
st.title("🧬 DNA Sequence Classifier Dashboard")

# === Sidebar Navigation ===
page = st.sidebar.radio("Go to", [
    "🔍 Classify DNA Sequence",
    "🧪 Generate Synthetic Data",
    "⚙️ Train Model",
    "🧠 Train Word2Vec",
    "📊 View Training History",
    "📂 View Sample Data"
])

# === Classify DNA Sequence ===
if page == "🔍 Classify DNA Sequence":
    st.header("🔍 Classify DNA Sequence")
    sequence_input = st.text_input("Enter DNA Sequence (ATGC only)", max_chars=200)

    if st.button("Classify"):
        if not sequence_input or not set(sequence_input.upper()).issubset({'A', 'T', 'G', 'C'}):
            st.warning("Invalid or empty DNA sequence. Use only A, T, G, C.")
        else:
            try:
                w2v_model, classifier = load_models()
                vector = sequence_to_vector(sequence_input.upper(), w2v_model).reshape(1, -1)
                prediction = classifier.predict(vector)[0][0]
                label = "Positive" if prediction > 0.5 else "Negative"
                st.success(f"Prediction: **{label}** ({prediction:.2f})")
            except Exception as e:
                st.error(f"Error: {e}")

# === Generate Synthetic Data ===
elif page == "🧪 Generate Synthetic Data":
    st.header("🧪 Generate Synthetic DNA Data")
    num_samples = st.number_input("Number of Samples", value=1000)
    length = st.number_input("Sequence Length", value=100)

    if st.button("Generate"):
        result = subprocess.run(
            ["python", "generatesynthdata.py"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            st.success("Synthetic DNA data generated!")
        else:
            st.error("Failed to generate data")
            st.text(result.stderr)

# === Train Model ===
elif page == "⚙️ Train Model":
    st.header("⚙️ Train CNN Classifier")
    if st.button("Train Model"):
        with st.spinner("Training model..."):
            process = subprocess.run(["python", "train.py"], capture_output=True, text=True,encoding="utf-8")
            st.text(process.stdout)
            if process.returncode == 0:
                st.success("Model trained and saved successfully!")
            else:
                st.error("Training failed.")
                st.text(process.stderr)

# === Train Word2Vec ===
elif page == "🧠 Train Word2Vec":
    st.header("🧠 Train Word2Vec Embeddings")
    if os.path.exists("synthetic_data.csv"):
        df = pd.read_csv("synthetic_data.csv")
        from utils import train_word2vec
        if st.button("Train Word2Vec"):
            with st.spinner("Training Word2Vec..."):
                train_word2vec(df["sequence"].values)
                st.success("Word2Vec model saved as `w2v.model`.")
    else:
        st.warning("Generate synthetic data first.")

# === View Training History ===
elif page == "📊 View Training History":
    st.header("📊 Training Accuracy Plot")
    try:
        import json
        with open('history.json', 'r') as f:
            content = f.read()
            if not content.strip():
                raise ValueError("history.json is empty.")
            history = json.loads(content)

        fig, ax = plt.subplots()
        ax.plot(history['accuracy'], label='Train Accuracy')
        ax.plot(history['val_accuracy'], label='Validation Accuracy')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Accuracy')
        ax.legend()
        st.pyplot(fig)

    except FileNotFoundError:
        st.error("No training history found. Please train the model first.")
    except json.JSONDecodeError:
        st.error("Training history is corrupted or empty. Please retrain the model.")
    except Exception as e:
        st.error(f"Error loading training history: {e}")

# === View Sample Data ===
elif page == "📂 View Sample Data":
    st.header("📂 Sample Synthetic Data")
    try:
        df = pd.read_csv("synthetic_data.csv")
        st.write(df.sample(5))
        st.info(f"Total Samples: {len(df)}")
        st.download_button("Download CSV", df.to_csv(index=False), file_name="synthetic_data.csv")
    except FileNotFoundError:
        st.warning("No synthetic data found. Please generate it first.")
