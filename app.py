import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

# Set page configuration
st.set_page_config(page_title="Sentiment Analysis of Real-time Flipkart Product Reviews", page_icon="🛍️")

# Download NLTK resources quietly on the server
@st.cache_resource
def download_nltk():
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

download_nltk()

# Load the saved model and vectorizer
@st.cache_resource
def load_models():
    if os.path.exists('sentiment_model.pkl') and os.path.exists('tfidf_vectorizer.pkl'):
        model = joblib.load('sentiment_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        return model, vectorizer
    else:
        return None, None

model, vectorizer = load_models()

# Initialize text processing tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def process_input(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text)).lower()
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return ' '.join(words)

# UI Layout
st.title("🛍️ Sentiment Analysis of Real-time Flipkart Product Reviews")
st.write("Analyze the sentiment of customer reviews in real-time.")

st.subheader("Model Performance")
st.write(f"Macro F1 Score: {0.7864:.4f}")
st.write(f"Weighted F1 Score: {0.8622:.4f}")

if model is None or vectorizer is None:
    st.error("Model files not found! Please run train_model.py first to generate the .pkl files.")
else:
    user_review = st.text_area("Paste a review here:", placeholder="Example: The product quality is good but delivery was late.")

    if st.button("Predict Sentiment", type="primary"):
        if not user_review.strip():
            st.warning("Please enter a review.")
        else:
            with st.spinner("Analyzing..."):
                cleaned_text = process_input(user_review)
                vectorized_text = vectorizer.transform([cleaned_text])
                prediction = model.predict(vectorized_text)[0]
                
                st.divider()
                if prediction == 1:
                    st.success("### Prediction: Positive Sentiment")
                else:
                    st.error("### Prediction: Negative Sentiment")