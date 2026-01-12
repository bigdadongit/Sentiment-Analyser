📊 Sentiment Analysis System

A lightweight NLP-based Sentiment Analysis web application that classifies text as Positive or Negative using Machine Learning.
Built with Python and Streamlit, and deployed live on Streamlit Community Cloud.

🌐 Live Demo

👉 Try the app here:
https://bigdad-sentiment-analyser.streamlit.app/

📸 Screenshots
<p align="center"> <img src="assets/home.png" width="380" /> <br /> <em>Home Screen</em> </p> <p align="center"> <img src="assets/result.png" width="380" /> <br /> <em>Sentiment Analysis Result</em> </p>
📝 Overview

The Sentiment Analysis System allows users to enter custom text or select sample examples and instantly receive sentiment predictions along with confidence scores.
The application automatically trains and loads the ML model on first run, making it fully cloud-compatible.

🧠 Tech Stack

Python

scikit-learn

NLP (TF-IDF, text preprocessing)

Streamlit

Plotly

⚙️ How It Works

User enters text or selects a sample

Text is cleaned and vectorized using TF-IDF

A Logistic Regression model predicts sentiment

Sentiment and confidence score are displayed

🚀 Deployment

Deployed on Streamlit Community Cloud

Automatically redeploys on every push to main

Model auto-trains if not found (no manual setup)

📌 Future Enhancements

Neutral sentiment classification

CSV upload for batch analysis

Word cloud visualization

REST API using FastAPI

⭐ Support

If you like this project, consider giving it a ⭐
It really helps!
