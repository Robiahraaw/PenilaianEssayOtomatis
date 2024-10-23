import streamlit as st
import numpy as np
import pandas as pd
import re
import unicodedata
import tensorflow_hub as hub
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.spatial.distance import cityblock
from scipy.stats import pearsonr
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Load the Universal Sentence Encoder
@st.cache_resource
def load_use_model():
    return hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

use_model = load_use_model()

# Text preprocessing function
def preprocess_text(text):
    # Trim whitespaces
    text = text.strip()
    # Remove accents
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    # Convert to lowercase
    text = text.lower()
    # Remove punctuations
    text = re.sub(r'[^\w\s]', '', text)
    # Remove stopwords and apply lemmatization using Sastrawi
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    
    stopword_factory = StopWordRemoverFactory()
    stopword_remover = stopword_factory.create_stop_word_remover()
    
    # Remove stopwords
    text = stopword_remover.remove(text)
    # Lemmatization
    text = stemmer.stem(text)
    
    return text

# Function to calculate various similarity metrics
def calculate_similarities(answer_key_emb, student_answer_emb):
    # Cosine Similarity
    cos_sim = cosine_similarity([answer_key_emb], [student_answer_emb])[0][0]
    # Euclidean Distance
    euc_dist = euclidean_distances([answer_key_emb], [student_answer_emb])[0][0]
    # Manhattan Distance
    man_dist = cityblock(answer_key_emb, student_answer_emb)
    # Pearson Correlation
    pearson_corr, _ = pearsonr(answer_key_emb, student_answer_emb)
    return cos_sim, euc_dist, man_dist, pearson_corr

# Streamlit App Layout
st.title("Implementasi Sistem Penilaian Esai Otomatis Menggunakan Metode Universal Sentence Encoder")

# Input form
with st.form("aes_form"):
    name = st.text_input("Nama")
    question = st.text_area("Soal")
    answer_key = st.text_area("Kunci Jawaban")
    student_answer = st.text_area("Jawaban Siswa")
    submitted = st.form_submit_button("Submit")

# Preprocessing and embedding
if submitted:
    # Preprocess the texts
    preprocessed_answer_key = preprocess_text(answer_key)
    preprocessed_student_answer = preprocess_text(student_answer)
    
    # Generate embeddings using USE
    answer_key_emb = use_model([preprocessed_answer_key])[0].numpy()
    student_answer_emb = use_model([preprocessed_student_answer])[0].numpy()
    
    # Calculate similarities
    cos_sim, euc_dist, man_dist, pearson_corr = calculate_similarities(answer_key_emb, student_answer_emb)
    
    # Apply threshold on Manhattan distance
    threshold = 19
    classification = 'Benar' if man_dist <= threshold else 'Salah'
    
    # Output B or S
    st.write(f"Jawaban Siswa **{classification}**")
    
    # Display results in a table
    results_df = pd.DataFrame({
        "Cosine Similarity": [cos_sim],
        "Euclidean Distance": [euc_dist],
        "Manhattan Distance": [man_dist],
        "Pearson Correlation": [pearson_corr]
    })
    
    st.write("Similarity Table:")
    st.table(results_df)