# Sentiment Analysis System

A production-ready sentiment analysis application utilizing Natural Language Processing and Machine Learning for automated text classification.

## Overview

This system implements an end-to-end sentiment analysis pipeline that classifies textual content into positive or negative sentiment categories. Built with industry-standard tools and algorithms, it provides reliable sentiment detection with high accuracy.

## Key Features

- **Real-time Analysis**: Instant sentiment classification of text input
- **High Accuracy**: 89.4% classification accuracy on test data
- **Professional UI**: Clean, intuitive web interface suitable for demonstrations
- **Robust Preprocessing**: Advanced NLP techniques for text normalization
- **Visual Analytics**: Interactive charts and confidence metrics
- **Balanced Classification**: Handles class imbalance for reliable predictions

## Technical Specifications

### Architecture

**Natural Language Processing Pipeline:**
1. Text normalization and cleaning
2. URL and mention removal
3. Special character filtering
4. Stop word elimination
5. Stemming (Porter Stemmer)

**Feature Engineering:**
- **Method**: TF-IDF Vectorization
- **Features**: 5,000 dimensions
- **N-grams**: Unigrams and Bigrams (1-2)

**Machine Learning Model:**
- **Algorithm**: Logistic Regression
- **Regularization**: L2 (Ridge)
- **Class Weighting**: Balanced
- **Convergence**: Maximum 1,000 iterations

### Dataset

- **Source**: Amazon Product Reviews
- **Size**: 3,150 samples
- **Distribution**: 
  - Positive (4-5 star): 87%
  - Negative (1-3 star): 13%
- **Format**: CSV (rating, date, variation, verified_reviews, feedback)

### Performance Metrics

- **Overall Accuracy**: 89.4%
- **Precision (Positive)**: 97%
- **Recall (Positive)**: 91%
- **F1-Score (Positive)**: 94%

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Setup Instructions

1. **Clone or download the project**
```bash
cd /path/to/sentiment
```

2. **Create virtual environment**
```bash
python3 -m venv venv
```

3. **Activate virtual environment**
```bash
# macOS/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

4. **Install dependencies**
```bash
pip install -r requirements.txt
```

5. **Train the model**
```bash
python sentiment_analyzer.py
```

6. **Launch the application**
```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## Usage

### Web Interface

1. Navigate to the application URL in your browser
2. Choose input method:
   - **Custom Text**: Enter your own text
   - **Sample Examples**: Select from predefined examples
3. Click "Analyze Sentiment" button
4. View results including:
   - Sentiment classification
   - Confidence score
   - Probability distribution
   - Visual analytics

### Programmatic Usage

```python
from sentiment_analyzer import SentimentAnalyzer

# Initialize and load model
analyzer = SentimentAnalyzer()
analyzer.load_model()

# Analyze text
text = "This product is absolutely amazing!"
prediction, probabilities = analyzer.predict(text)

# Get results
sentiment = "Positive" if prediction == 1 else "Negative"
confidence = probabilities[int(prediction)] * 100

print(f"Sentiment: {sentiment}")
print(f"Confidence: {confidence:.2f}%")
```

## Project Structure

```
sentiment/
├── app.py                      # Streamlit web application
├── sentiment_analyzer.py       # Core ML model and training
├── requirements.txt            # Python dependencies
├── README.md                   # Documentation
├── run.sh                      # Quick start script
├── .gitignore                  # Git ignore rules
├── data/
│   ├── amazon_reviews.csv      # Training dataset
│   └── twitter.csv             # Alternative dataset
├── models/
│   ├── model.pkl               # Trained classifier
│   └── vectorizer.pkl          # TF-IDF vectorizer
└── venv/                       # Virtual environment (excluded)
```

## Technology Stack

| Component | Technology |
|-----------|------------|
| **Frontend** | Streamlit 1.51.0 |
| **ML Framework** | scikit-learn 1.7.2 |
| **NLP Library** | NLTK 3.9.2 |
| **Data Processing** | Pandas 2.3.3, NumPy 2.3.5 |
| **Visualization** | Plotly 6.5.0 |
| **Language** | Python 3.13 |

## Model Training

The model is trained on Amazon product reviews with the following process:

1. **Data Loading**: CSV parsing with proper encoding
2. **Label Creation**: Convert ratings (1-5) to binary sentiment
3. **Text Preprocessing**: Clean and normalize text
4. **Feature Extraction**: TF-IDF vectorization
5. **Model Training**: Logistic Regression with balanced weights
6. **Evaluation**: Classification metrics on test set
7. **Model Persistence**: Pickle serialization

To retrain the model:
```bash
python sentiment_analyzer.py
```

## Dependencies

```
streamlit        # Web framework
pandas           # Data manipulation
numpy            # Numerical operations
scikit-learn     # Machine learning
nltk             # Natural language processing
plotly           # Interactive visualizations
```

## Deployment Considerations

For production deployment:

1. **Environment Variables**: Configure via `.env` file
2. **Model Versioning**: Implement model version control
3. **API Integration**: Wrap in REST API (FastAPI/Flask)
4. **Scalability**: Consider containerization (Docker)
5. **Monitoring**: Add logging and performance metrics
6. **Security**: Implement input validation and rate limiting

## Academic Use

This project is designed for academic presentations and demonstrations. When presenting:

- Highlight the 89.4% accuracy metric
- Explain the preprocessing pipeline
- Demonstrate the class balancing technique
- Show real-time predictions
- Discuss the TF-IDF feature extraction
- Present the confusion matrix and metrics

## Limitations

- Binary classification only (positive/negative)
- English language text only
- Trained on product review domain
- May require retraining for other domains
- Limited context understanding

## Future Enhancements

- Multi-class sentiment (neutral category)
- Multi-language support
- Deep learning models (LSTM, BERT)
- Real-time training updates
- Batch processing capability
- REST API endpoint
- Enhanced visualization dashboard

## License

This project is created for educational purposes.

## Contact

For questions or collaboration, please contact through your institution.

---

**Note**: This is an academic project demonstrating NLP and ML concepts for sentiment analysis.
