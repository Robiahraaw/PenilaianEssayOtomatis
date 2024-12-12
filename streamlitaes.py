import streamlit as st
import numpy as np
import pandas as pd
import re
import unicodedata
import tensorflow_hub as hub
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
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
    
    # Calculate Cosine Similarity
    cos_sim = cosine_similarity([answer_key_emb], [student_answer_emb])[0][0]
    
    # Calculate the score by multiplying Cosine Similarity by 10
    student_score = cos_sim * 10
    
    # Display the student's score
    st.write(f"**Bobot Jawaban Siswa: {student_score:.2f}**")
