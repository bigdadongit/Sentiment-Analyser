#!/bin/bash

echo "======================================"
echo "  Sentiment Analysis System"
echo "======================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

# Check if dependencies are installed
if ! python -c "import streamlit" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Check if model exists
if [ ! -f "models/model.pkl" ]; then
    echo "Training model (this may take a few minutes)..."
    python sentiment_analyzer.py
fi

echo ""
echo "======================================"
echo "  Starting Application..."
echo "======================================"
echo ""
streamlit run app.py
