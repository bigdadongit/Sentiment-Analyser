import pandas as pd
import numpy as np
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

class SentimentAnalyzer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize and remove stopwords
        words = text.split()
        words = [self.stemmer.stem(word) for word in words if word not in self.stop_words and len(word) > 2]
        
        return ' '.join(words)
    
    def train(self, data_path):
        """Train the sentiment analysis model"""
        print("Loading dataset...")
        df = pd.read_csv(data_path, encoding='latin-1')
        
        # Check if it's Amazon reviews or Twitter data
        if 'rating' in df.columns:
            # Amazon reviews: convert ratings to sentiment (1-3 = negative/0, 4-5 = positive/1)
            df['sentiment'] = df['rating'].apply(lambda x: 1 if x >= 4 else 0)
            df['text'] = df['verified_reviews']
        else:
            # Twitter data
            df.columns = ['id', 'sentiment', 'text']
        
        print(f"Dataset loaded: {len(df)} reviews/tweets")
        print(f"Sentiment distribution:\n{df['sentiment'].value_counts()}")
        
        # Preprocess texts
        print("Preprocessing texts...")
        df['cleaned_text'] = df['text'].apply(self.preprocess_text)
        
        # Remove empty texts
        df = df[df['cleaned_text'].str.len() > 0]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['cleaned_text'], df['sentiment'], test_size=0.2, random_state=42
        )
        
        print("Vectorizing texts...")
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        print("Training model...")
        self.model.fit(X_train_vec, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nModel Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
        
        return accuracy
    
    def predict(self, text):
        """Predict sentiment for a given text"""
        cleaned_text = self.preprocess_text(text)
        if not cleaned_text:
            return 0, 0.5
        
        text_vec = self.vectorizer.transform([cleaned_text])
        prediction = self.model.predict(text_vec)[0]
        probability = self.model.predict_proba(text_vec)[0]
        
        return prediction, probability
    
    def save_model(self, vectorizer_path='models/vectorizer.pkl', model_path='models/model.pkl'):
        """Save the trained model and vectorizer"""
        import os
        os.makedirs('models', exist_ok=True)
        
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        print(f"Model saved to {model_path}")
        print(f"Vectorizer saved to {vectorizer_path}")
    
    def load_model(self, vectorizer_path='models/vectorizer.pkl', model_path='models/model.pkl'):
        """Load a pre-trained model and vectorizer"""
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        print("Model loaded successfully")

if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    
    # Train the model using Amazon reviews dataset
    accuracy = analyzer.train('data/amazon_reviews.csv')
    
    # Save the model
    analyzer.save_model()
    
    # Test predictions
    test_texts = [
        "I love this product! It's amazing!",
        "This is terrible. I hate it.",
        "Not sure how I feel about this.",
        "Best day ever! So happy!"
    ]
    
    print("\n" + "="*50)
    print("Testing predictions:")
    print("="*50)
    
    for text in test_texts:
        prediction, probability = analyzer.predict(text)
        sentiment = "Positive" if prediction == 1 else "Negative"
        confidence = probability[int(prediction)] * 100
        print(f"\nText: {text}")
        print(f"Sentiment: {sentiment} (Confidence: {confidence:.2f}%)")
