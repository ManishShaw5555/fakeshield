import streamlit as st
import requests
from bs4 import BeautifulSoup
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---- Configuration ---- #
MODEL_PATH = "cnn_fake_news_model.h5"
TOKENIZER_PATH = "tokenizer.pkl"
MAX_SEQUENCE_LENGTH = 300

# ---- Load Model & Tokenizer ---- #
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

@st.cache_resource
def load_tokenizer():
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    return tokenizer

model = load_model()
tokenizer = load_tokenizer()

# ---- Utility: Get article text from URL ---- #
def extract_text_from_url(url):
    try:
        page = requests.get(url, timeout=5)
        soup = BeautifulSoup(page.content, "html.parser")
        paragraphs = soup.find_all('p')
        article_text = ' '.join(p.get_text() for p in paragraphs)
        return article_text.strip()
    except Exception as e:
        st.error(f"⚠️ Error fetching URL: {e}")
        return ""

# ---- Text Preprocessing ---- #
def preprocess_text(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    return padded

# ---- Prediction ---- #
def predict(text):
    processed = preprocess_text(text)
    prob = model.predict(processed)[0][0]
    return "FAKE NEWS" if prob >= 0.5 else "REAL NEWS", float(prob)

# ---- UI: Streamlit Layout ---- #
st.set_page_config(page_title="🛡️ FactShield - Fake Article Detector", layout="centered")

st.markdown("""
    <style>
        .title {
            text-align: center;
            font-size: 48px;
            font-weight: bold;
            color: #00B8A9;
            margin-bottom: 20px;
        }
        .prediction-box {
            border-radius: 15px;
            padding: 20px;
            background-color: #F9F9F9;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        }
        .footer {
            text-align: center;
            font-size: 14px;
            margin-top: 50px;
            color: gray;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🛡️ FactShield - Fake Article Detector</div>', unsafe_allow_html=True)

# --- Input Section --- #
input_mode = st.radio("Choose input method:", ["📝 Paste Article", "🌐 Article URL"])

article_text = ""
if input_mode == "📝 Paste Article":
    article_text = st.text_area("Paste article content here:", height=200)
else:
    url = st.text_input("Enter article URL:")
    if url:
        article_text = extract_text_from_url(url)
        st.success("✅ Article text extracted from URL.")

# --- Predict Button --- #
if article_text and st.button("🔍 Analyze Article"):
    with st.spinner("Analyzing..."):
        prediction, confidence = predict(article_text)

    st.markdown(f"""
        <div class="prediction-box">
            <h3>🧠 Prediction:</h3>
            <p style="font-size:28px; color:{'red' if prediction=='FAKE NEWS' else 'green'};"><strong>{prediction}</strong></p>
            <p>Confidence Score: <code>{confidence:.2f}</code></p>
        </div>
    """, unsafe_allow_html=True)

# --- Footer --- #
st.markdown('<div class="footer">Made with ❤️ using Streamlit | © 2025 FactShield</div>', unsafe_allow_html=True)
