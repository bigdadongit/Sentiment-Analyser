import pandas as pd
import numpy as np
import re
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download required NLTK data
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)


class SentimentAnalyzer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.model = LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced"
        )
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words("english"))

    def preprocess_text(self, text):
        text = str(text).lower()
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)
        text = re.sub(r"@\w+|#\w+", "", text)
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()

        words = text.split()
        words = [
            self.stemmer.stem(word)
            for word in words
            if word not in self.stop_words and len(word) > 2
        ]
        return " ".join(words)

    def train(self, data_path):
        df = pd.read_csv(data_path, encoding="latin-1")

        if "rating" in df.columns:
            df["sentiment"] = df["rating"].apply(lambda x: 1 if x >= 4 else 0)
            df["text"] = df["verified_reviews"]
        else:
            df.columns = ["id", "sentiment", "text"]

        df["cleaned_text"] = df["text"].apply(self.preprocess_text)
        df = df[df["cleaned_text"].str.len() > 0]

        X_train, X_test, y_train, y_test = train_test_split(
            df["cleaned_text"],
            df["sentiment"],
            test_size=0.2,
            random_state=42,
        )

        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)

        self.model.fit(X_train_vec, y_train)

        y_pred = self.model.predict(X_test_vec)
        acc = accuracy_score(y_test, y_pred)

        print(f"Model trained with accuracy: {acc:.4f}")
        return acc

    def predict(self, text):
        cleaned = self.preprocess_text(text)
        if not cleaned:
            return 0, [0.5, 0.5]

        vec = self.vectorizer.transform([cleaned])
        pred = self.model.predict(vec)[0]
        prob = self.model.predict_proba(vec)[0]

        return pred, prob

    def save_model(self, vectorizer_path, model_path):
        os.makedirs("models", exist_ok=True)
        with open(vectorizer_path, "wb") as f:
            pickle.dump(self.vectorizer, f)
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)

    def load_model(self, vectorizer_path, model_path):
        with open(vectorizer_path, "rb") as f:
            self.vectorizer = pickle.load(f)
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)


#  STREAMLIT-SAFE LOADER
def get_analyzer():
    analyzer = SentimentAnalyzer()

    vectorizer_path = "models/vectorizer.pkl"
    model_path = "models/model.pkl"

    if not os.path.exists(vectorizer_path) or not os.path.exists(model_path):
        print("Model not found. Training...")
        analyzer.train("data/amazon_reviews.csv")
        analyzer.save_model(vectorizer_path, model_path)
    else:
        analyzer.load_model(vectorizer_path, model_path)

    return analyzer


if __name__ == "__main__":
    analyzer = get_analyzer()

    samples = [
        "I love this product",
        "Worst purchase ever",
        "Not bad, could be better",
    ]

    for text in samples:
        pred, prob = analyzer.predict(text)
        label = "Positive" if pred == 1 else "Negative"
        confidence = prob[pred] * 100
        print(f"{text} → {label} ({confidence:.2f}%)")

