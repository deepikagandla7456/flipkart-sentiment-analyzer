import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
import joblib
import os

# 1. Download necessary NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

def run_pipeline():
    print("Loading dataset...")
    
    # Build the exact path to the CSV file
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CSV_PATH = os.path.join(BASE_DIR, 'reviews_badminton', 'data.csv')
    
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print(f"\n ERROR: Could not find the file at {CSV_PATH}")
        print("Ensure the 'reviews_badminton' folder is in the same directory as this script.")
        return

    # EXACT COLUMN NAMES APPLIED HERE
    TEXT_COLUMN = 'Review text'  
    RATING_COLUMN = 'Ratings'     

    # Handle missing values using the correct column names
    df.dropna(subset=[TEXT_COLUMN, RATING_COLUMN], inplace=True)

    # Convert Rating to Sentiment (Objective: positive/negative)
    # Ratings 4-5 are Positive (1), Ratings 1-3 are Negative (0)
    df['Sentiment'] = df[RATING_COLUMN].apply(lambda x: 1 if float(x) > 3 else 0)

    # 2. Data Preprocessing setup
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    def clean_text(text):
        # Text Cleaning: Remove special characters and punctuation
        text = re.sub(r'[^a-zA-Z\s]', '', str(text)).lower()
        # Text Normalization: Lemmatization and removing stopwords
        words = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
        return ' '.join(words)

    print("Cleaning text data... this may take a few seconds.")
    df['Cleaned_Text'] = df[TEXT_COLUMN].apply(clean_text)

    # 3. Text Embedding (TF-IDF Feature Extraction)
    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X = vectorizer.fit_transform(df['Cleaned_Text'])
    y = df['Sentiment']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 4. Modeling Approach (Logistic Regression)
    print("Training model...")
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train, y_train)

    # 5. Evaluation (F1-Score)
    y_pred = model.predict(X_test)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    print("\n--- Model Evaluation ---")
    print(f"Macro F1-Score: {f1_macro:.4f}")
    print(f"Weighted F1-Score: {f1_weighted:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Save artifacts for deployment
    joblib.dump(model, 'sentiment_model.pkl')
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
    print("\n Artifacts saved successfully: 'sentiment_model.pkl' and 'tfidf_vectorizer.pkl'")

if __name__ == "__main__":
    run_pipeline()