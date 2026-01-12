import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sentiment_analyzer import get_analyzer

# ==================================================
# Page Configuration
# ==================================================
st.set_page_config(
    page_title="Sentiment Analysis System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# GLOBAL LIGHT THEME + FIXED BUTTON STYLES
# ==================================================
st.markdown("""
<style>
/* Force light background */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div:first-child {
    background-color: #ffffff !important;
}

/* Force readable text */
.stMarkdown, p, span, div, label, h1, h2, h3, h4, h5, h6 {
    color: #000000 !important;
}

/* ===== BUTTON FIX (IMPORTANT) ===== */
.stButton > button {
    background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%) !important;
    color: #ffffff !important;
    font-weight: 600;
    border-radius: 8px;
    border: none;
    padding: 0.8rem 2.5rem;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #1d4ed8 0%, #1e3a8a 100%) !important;
    color: #ffffff !important;
}

/* Result boxes */
.positive-box {
    background: linear-gradient(135deg, #f0fdf4, #dcfce7);
    border-left: 5px solid #22c55e;
    border-radius: 12px;
    padding: 28px;
    margin: 24px 0;
}

.negative-box {
    background: linear-gradient(135deg, #fef2f2, #fee2e2);
    border-left: 5px solid #ef4444;
    border-radius: 12px;
    padding: 28px;
    margin: 24px 0;
}

/* Metric cards */
.metric-card {
    background: #ffffff;
    border-radius: 10px;
    padding: 18px;
    text-align: center;
    border: 1px solid #e5e7eb;
}

/* Sidebar status */
.status-active {
    background: linear-gradient(135deg, #f0fdf4, #dcfce7);
    padding: 12px;
    border-radius: 6px;
    border-left: 3px solid #22c55e;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# Load Model (AUTO-TRAIN SAFE)
# ==================================================
@st.cache_resource
def load_model():
    return get_analyzer()

with st.spinner("Loading sentiment model..."):
    analyzer = load_model()

# ==================================================
# Header
# ==================================================
st.markdown("""
<div style="text-align:center; padding:30px 0 40px 0;">
    <h1 style="font-size:48px; margin-bottom:10px;">Sentiment Analysis System</h1>
    <p style="font-size:18px;">Advanced NLP-Based Text Sentiment Classification</p>
</div>
""", unsafe_allow_html=True)

# ==================================================
# Sidebar
# ==================================================
with st.sidebar:
    st.markdown("### System Status")
    st.markdown("""
        <div class="status-active">
            <strong style="color:#16a34a;">●</strong>
            <span style="font-weight:500;">Model Active</span>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("### How to Use")
    st.markdown("""
    1. Choose input method  
    2. Enter or select text  
    3. Click **Analyze Sentiment**
    """)

# ==================================================
# Input Section
# ==================================================
_, center_col, _ = st.columns([1, 4, 1])

with center_col:
    input_method = st.radio(
        "Input Method",
        ["Custom Text", "Sample Examples"],
        horizontal=True
    )

    if input_method == "Custom Text":
        user_text = st.text_area(
            "Enter text",
            height=180,
            placeholder="Type or paste text here...",
            label_visibility="collapsed"
        )
    else:
        samples = {
            "Positive Review": "I absolutely love this product. Best purchase ever!",
            "Negative Review": "Worst experience ever. Completely disappointed.",
            "Mixed Feedback": "The product is okay but has some issues.",
            "Positive Feedback": "Amazing service and excellent quality.",
            "Negative Feedback": "Terrible support and poor quality."
        }
        choice = st.selectbox("Choose sample", list(samples.keys()))
        user_text = st.text_area(
            "Sample text",
            value=samples[choice],
            height=180,
            label_visibility="collapsed"
        )

    analyze_btn = st.button("Analyze Sentiment", use_container_width=True)

# ==================================================
# Analysis Results
# ==================================================
if analyze_btn:
    if not user_text.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing sentiment..."):
            prediction, probabilities = analyzer.predict(user_text)
            sentiment = "Positive" if prediction == 1 else "Negative"
            confidence = probabilities[prediction] * 100

        st.markdown("## Analysis Results")

        col1, col2 = st.columns(2)

        with col1:
            if sentiment == "Positive":
                st.markdown(f"""
                <div class="positive-box">
                    <h2>POSITIVE SENTIMENT</h2>
                    <p>Confidence: <strong>{confidence:.2f}%</strong></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="negative-box">
                    <h2>NEGATIVE SENTIMENT</h2>
                    <p>Confidence: <strong>{confidence:.2f}%</strong></p>
                </div>
                """, unsafe_allow_html=True)

            prob_df = pd.DataFrame({
                "Sentiment": ["Negative", "Positive"],
                "Probability (%)": [probabilities[0]*100, probabilities[1]*100]
            })
            st.bar_chart(prob_df.set_index("Sentiment"))

        with col2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence,
                gauge={"axis": {"range": [0, 100]}},
                title={"text": "Confidence"}
            ))
            st.plotly_chart(fig, use_container_width=True)

# ==================================================
# Footer
# ==================================================
st.markdown("""
<div style="text-align:center; margin-top:50px; color:#94a3b8;">
    © 2025 Sentiment Analysis System
</div>
""", unsafe_allow_html=True)
