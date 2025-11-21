import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sentiment_analyzer import SentimentAnalyzer
import os

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Force light theme
st.markdown("""
    <style>
    /* Force white background everywhere */
    .stApp {
        background-color: #ffffff !important;
    }
    
    [data-testid="stAppViewContainer"] {
        background-color: #ffffff !important;
    }
    
    [data-testid="stHeader"] {
        background-color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] > div:first-child {
        background-color: #ffffff !important;
    }
    
    /* Force black text */
    .stMarkdown, p, span, div, label, h1, h2, h3, h4, h5, h6 {
        color: #000000 !important;
    }
    </style>
""", unsafe_allow_html=True)

    # Custom CSS for Clean Professional UI with Light Tones
st.markdown("""
    <style>
    /* Global Styles */
    .main {
        padding: 2rem;
        background: #ffffff;
        min-height: 100vh;
    }
    
    .stTextArea textarea {
        font-size: 16px;
        font-family: 'Inter', 'Segoe UI', sans-serif;
        border: 2px solid #e5e7eb;
        border-radius: 8px;
        padding: 12px;
        transition: all 0.3s ease;
        background: #ffffff;
        color: #000000;
    }
    
    .stTextArea textarea:focus {
        border-color: #93c5fd;
        box-shadow: 0 0 0 3px rgba(147, 197, 253, 0.2);
    }
    
    /* Header Styling */
    h1 {
        color: #000000;
        font-weight: 700;
        letter-spacing: -1px;
    }
    
    h2, h3 {
        color: #000000;
        font-weight: 600;
    }
    
    /* Result Cards - Light Tones */
    .positive-box {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border-left: 5px solid #4ade80;
        border-radius: 12px;
        padding: 32px;
        margin: 24px 0;
        box-shadow: 0 4px 16px rgba(74, 222, 128, 0.1);
    }
    
    .negative-box {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border-left: 5px solid #f87171;
        border-radius: 12px;
        padding: 32px;
        margin: 24px 0;
        box-shadow: 0 4px 16px rgba(248, 113, 113, 0.1);
    }
    
    /* Metric Cards */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        border: 1px solid #e5e7eb;
    }
    
    /* Buttons - Light Blue */
    .stButton>button {
        background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 100%);
        color: white;
        border-radius: 8px;
        padding: 0.75rem 3rem;
        font-weight: 600;
        font-size: 16px;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(96, 165, 250, 0.25);
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.35);
        transform: translateY(-2px);
    }
    
    /* Professional Typography */
    .result-title {
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 16px;
        color: #000000;
        letter-spacing: -0.5px;
    }
    
    .result-description {
        font-size: 17px;
        line-height: 1.7;
        color: #000000;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 40px 0 20px 0;
        color: #6b7280;
        border-top: 1px solid #e5e7eb;
        margin-top: 60px;
    }
    
    /* Radio buttons */
    .stRadio > label {
        font-weight: 500;
        color: #000000;
    }
    
    /* Selectbox */
    .stSelectbox > label {
        font-weight: 500;
        color: #000000;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: #ffffff;
    }
    
    /* Info box styling */
    .info-box {
        background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%);
        border-left: 4px solid #60a5fa;
        border-radius: 8px;
        padding: 16px;
        margin: 16px 0;
    }
    
    .status-active {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        padding: 12px;
        border-radius: 6px;
        border-left: 3px solid #4ade80;
    }
    
    .status-inactive {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        padding: 12px;
        border-radius: 6px;
        border-left: 3px solid #f87171;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_analyzer():
    """Load the sentiment analyzer model"""
    analyzer = SentimentAnalyzer()
    
    if os.path.exists('models/model.pkl') and os.path.exists('models/vectorizer.pkl'):
        analyzer.load_model()
    else:
        st.warning("Model not found. Please train the model first by running sentiment_analyzer.py")
        return None
    
    return analyzer

def create_gauge_chart(confidence, sentiment):
    """Create a gauge chart for confidence visualization"""
    color = "#28a745" if sentiment == "Positive" else "#dc3545"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#ffebee'},
                {'range': [50, 75], 'color': '#fff3e0'},
                {'range': [75, 100], 'color': '#e8f5e9'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig

def main():
    # Professional Header with Black Text
    st.markdown("""
        <div style='text-align: center; padding: 30px 0 40px 0;'>
            <h1 style='color: #000000; font-size: 48px; margin-bottom: 12px; font-weight: 700;'>
                Sentiment Analysis System
            </h1>
            <p style='color: #000000; font-size: 20px; margin: 0; font-weight: 400;'>
                Advanced NLP-Based Text Sentiment Classification
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar - Minimal
    with st.sidebar:
        st.markdown("### System Status")
        if os.path.exists('models/model.pkl'):
            st.markdown("""
                <div class='status-active'>
                    <strong style='color: #22c55e;'>●</strong> 
                    <span style='color: #16a34a; font-weight: 500;'>Model Active</span>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class='status-inactive'>
                    <strong style='color: #ef4444;'>●</strong> 
                    <span style='color: #dc2626; font-weight: 500;'>Model Not Found</span>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
        st.markdown("### How to Use")
        st.markdown("""
        <div class='info-box'>
            <div style='margin-bottom: 14px;'>
                <strong style='color: #000000; font-size: 15px;'>Step 1:</strong>
                <p style='margin: 4px 0 0 0; color: #000000; line-height: 1.6;'>
                    Select your input method - Custom Text or Sample Examples
                </p>
            </div>
            <div style='margin-bottom: 14px;'>
                <strong style='color: #000000; font-size: 15px;'>Step 2:</strong>
                <p style='margin: 4px 0 0 0; color: #000000; line-height: 1.6;'>
                    Enter or select the text you want to analyze
                </p>
            </div>
            <div>
                <strong style='color: #000000; font-size: 15px;'>Step 3:</strong>
                <p style='margin: 4px 0 0 0; color: #000000; line-height: 1.6;'>
                    Click "Analyze Sentiment" and view the results
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Load analyzer
    analyzer = load_analyzer()
    
    if analyzer is None:
        st.error("Please train the model first by running: `python sentiment_analyzer.py`")
        return
    
    # Main content - Centered and clean
    st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
    
    # Single column centered layout
    col_spacer1, col_main, col_spacer2 = st.columns([0.5, 3, 0.5])
    
    with col_main:
        # Text input options
        input_method = st.radio(
            "Input Method:",
            ["Custom Text", "Sample Examples"],
            horizontal=True
        )
        
        if input_method == "Custom Text":
            user_input = st.text_area(
                "Enter Text for Analysis:",
                height=200,
                placeholder="Type or paste your text here (reviews, feedback, comments, etc.)",
                label_visibility="collapsed"
            )
        else:
            example_texts = {
                "Positive Product Review": "I absolutely love this product! It exceeded all my expectations. Best purchase ever!",
                "Negative Service Review": "This is the worst experience I've ever had. Completely disappointed and frustrated.",
                "Mixed Product Feedback": "The product is okay, some features are good but others need improvement.",
                "Positive Customer Feedback": "Excellent service and quality. Highly recommend to everyone.",
                "Negative Experience": "Poor quality and terrible customer support. Would not recommend.",
            }
            
            selected_example = st.selectbox("Select Sample Text:", list(example_texts.keys()))
            user_input = st.text_area(
                "Sample Text:",
                value=example_texts[selected_example],
                height=200,
                label_visibility="collapsed"
            )
        
        st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
        analyze_button = st.button("Analyze Sentiment", type="primary", use_container_width=True)
    
    # Analysis results
    if analyze_button and user_input:
        with st.spinner("Processing sentiment analysis..."):
            prediction, probabilities = analyzer.predict(user_input)
            sentiment = "Positive" if prediction == 1 else "Negative"
            confidence = probabilities[int(prediction)] * 100
            
            st.markdown("<div style='margin-top: 40px;'></div>", unsafe_allow_html=True)
            st.markdown("### Analysis Results")
            
            # Results layout
            res_col1, res_col2 = st.columns([1, 1])
            
            with res_col1:
                # Sentiment result
                if sentiment == "Positive":
                    st.markdown(f"""
                        <div class="positive-box">
                            <p class="result-title" style="color: #22c55e; margin: 0;">
                                POSITIVE SENTIMENT
                            </p>
                            <p class="result-description" style="margin-top: 12px; margin-bottom: 0;">
                                The analyzed text exhibits positive sentiment characteristics 
                                with a confidence score of <strong>{confidence:.2f}%</strong>.
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="negative-box">
                            <p class="result-title" style="color: #f87171; margin: 0;">
                                NEGATIVE SENTIMENT
                            </p>
                            <p class="result-description" style="margin-top: 12px; margin-bottom: 0;">
                                The analyzed text exhibits negative sentiment characteristics 
                                with a confidence score of <strong>{confidence:.2f}%</strong>.
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Probability breakdown
                st.markdown("#### Probability Distribution")
                prob_df = pd.DataFrame({
                    'Sentiment': ['Negative', 'Positive'],
                    'Probability (%)': [probabilities[0] * 100, probabilities[1] * 100]
                })
                
                st.bar_chart(prob_df.set_index('Sentiment'))
            
            with res_col2:
                # Gauge chart
                st.plotly_chart(create_gauge_chart(confidence, sentiment), use_container_width=False)
                
                # Analytical Reasoning - WHY this conclusion
                st.markdown("""
                    <div style='text-align: center; margin: 20px 0 16px 0;'>
                        <h4 style='color: #000000; font-size: 20px; font-weight: 600; margin: 0;'>
                            💡 Analysis Insights
                        </h4>
                    </div>
                """, unsafe_allow_html=True)
                
                # Analyze the text for reasoning
                text_lower = user_input.lower()
                
                # Positive indicators
                positive_words = ['love', 'great', 'excellent', 'amazing', 'wonderful', 'best', 
                                'fantastic', 'good', 'happy', 'perfect', 'recommend', 'awesome',
                                'outstanding', 'superb', 'brilliant', 'exceptional']
                
                # Negative indicators
                negative_words = ['hate', 'bad', 'worst', 'terrible', 'awful', 'poor', 
                                'disappointing', 'disappointed', 'horrible', 'useless', 'waste',
                                'never', 'don\'t', 'not', 'no', 'issue', 'problem', 'complaint']
                
                # Intensifiers
                intensifiers = ['very', 'extremely', 'absolutely', 'completely', 'totally', 'really']
                
                # Count indicators
                pos_count = sum(1 for word in positive_words if word in text_lower)
                neg_count = sum(1 for word in negative_words if word in text_lower)
                intensifier_count = sum(1 for word in intensifiers if word in text_lower)
                
                # Detect exclamation marks (indicate strong emotion)
                exclamations = user_input.count('!')
                
                # Word count
                word_count = len(user_input.split())
                
                # Build reasoning with attractive formatting
                reasoning_items = []
                
                if sentiment == "Positive":
                    # Main finding
                    reasoning_items.append(f"""
                        <div style='background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%); 
                        padding: 14px; border-radius: 8px; margin-bottom: 12px; border-left: 4px solid #22c55e;'>
                            <strong style='color: #166534; font-size: 15px;'>✓ Positive Language</strong>
                            <p style='margin: 6px 0 0 0; color: #000000; font-size: 14px;'>
                                Detected {pos_count} positive indicator(s) in the text
                            </p>
                        </div>
                    """)
                    
                    if exclamations > 0:
                        reasoning_items.append(f"""
                            <div style='background: #fef3c7; padding: 14px; border-radius: 8px; 
                            margin-bottom: 12px; border-left: 4px solid #f59e0b;'>
                                <strong style='color: #92400e; font-size: 15px;'>⚡ Enthusiasm</strong>
                                <p style='margin: 6px 0 0 0; color: #000000; font-size: 14px;'>
                                    {exclamations} exclamation mark(s) show strong emotion
                                </p>
                            </div>
                        """)
                    
                    if intensifier_count > 0:
                        reasoning_items.append(f"""
                            <div style='background: #ddd6fe; padding: 14px; border-radius: 8px; 
                            margin-bottom: 12px; border-left: 4px solid #7c3aed;'>
                                <strong style='color: #5b21b6; font-size: 15px;'>🔥 Emphasis</strong>
                                <p style='margin: 6px 0 0 0; color: #000000; font-size: 14px;'>
                                    {intensifier_count} intensifying word(s) strengthen sentiment
                                </p>
                            </div>
                        """)
                    
                    if neg_count > 0:
                        reasoning_items.append(f"""
                            <div style='background: #fee2e2; padding: 14px; border-radius: 8px; 
                            margin-bottom: 12px; border-left: 4px solid #ef4444;'>
                                <strong style='color: #991b1b; font-size: 15px;'>⚠️ Mixed Signals</strong>
                                <p style='margin: 6px 0 0 0; color: #000000; font-size: 14px;'>
                                    {neg_count} negative word(s) present but outweighed
                                </p>
                            </div>
                        """)
                    
                else:  # Negative
                    # Main finding
                    reasoning_items.append(f"""
                        <div style='background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); 
                        padding: 14px; border-radius: 8px; margin-bottom: 12px; border-left: 4px solid #ef4444;'>
                            <strong style='color: #991b1b; font-size: 15px;'>✗ Negative Language</strong>
                            <p style='margin: 6px 0 0 0; color: #000000; font-size: 14px;'>
                                Detected {neg_count} negative indicator(s) in the text
                            </p>
                        </div>
                    """)
                    
                    if intensifier_count > 0:
                        reasoning_items.append(f"""
                            <div style='background: #fef3c7; padding: 14px; border-radius: 8px; 
                            margin-bottom: 12px; border-left: 4px solid #f59e0b;'>
                                <strong style='color: #92400e; font-size: 15px;'>🔥 Emphasis</strong>
                                <p style='margin: 6px 0 0 0; color: #000000; font-size: 14px;'>
                                    {intensifier_count} intensifying word(s) strengthen sentiment
                                </p>
                            </div>
                        """)
                    
                    if pos_count > 0:
                        reasoning_items.append(f"""
                            <div style='background: #dcfce7; padding: 14px; border-radius: 8px; 
                            margin-bottom: 12px; border-left: 4px solid #22c55e;'>
                                <strong style='color: #166534; font-size: 15px;'>⚠️ Mixed Signals</strong>
                                <p style='margin: 6px 0 0 0; color: #000000; font-size: 14px;'>
                                    {pos_count} positive word(s) present but outweighed
                                </p>
                            </div>
                        """)
                
                # Confidence card
                confidence_color = "#22c55e" if confidence > 80 else "#3b82f6" if confidence > 60 else "#f59e0b"
                confidence_bg = "#dcfce7" if confidence > 80 else "#dbeafe" if confidence > 60 else "#fef3c7"
                confidence_icon = "🎯" if confidence > 80 else "📊" if confidence > 60 else "🔍"
                
                reasoning_items.append(f"""
                    <div style='background: {confidence_bg}; padding: 14px; border-radius: 8px; 
                    margin-bottom: 12px; border-left: 4px solid {confidence_color};'>
                        <strong style='color: #000000; font-size: 15px;'>{confidence_icon} Confidence</strong>
                        <p style='margin: 6px 0 0 0; color: #000000; font-size: 14px;'>
                            {confidence:.1f}% - {('High' if confidence > 80 else 'Moderate' if confidence > 60 else 'Low')} certainty level
                        </p>
                    </div>
                """)
                
                # Probability margin
                prob_diff = abs(probabilities[1] - probabilities[0]) * 100
                reasoning_items.append(f"""
                    <div style='background: #e0e7ff; padding: 14px; border-radius: 8px; 
                    margin-bottom: 12px; border-left: 4px solid #6366f1;'>
                        <strong style='color: #3730a3; font-size: 15px;'>📈 Probability Gap</strong>
                        <p style='margin: 6px 0 0 0; color: #000000; font-size: 14px;'>
                            {prob_diff:.1f}% separation between sentiments
                        </p>
                    </div>
                """)
                
                # Combine all reasoning items
                reasoning_html = "".join(reasoning_items)
                
                # Display all insights at once
                for item_html in reasoning_items:
                    st.markdown(item_html, unsafe_allow_html=True)
                
                # Additional metrics
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f"""
                        <div class='metric-card'>
                            <p style='color: #6b7280; margin: 0; font-size: 12px;'>CONFIDENCE</p>
                            <h3 style='color: #000000; margin: 8px 0 0 0;'>{confidence:.2f}%</h3>
                        </div>
                    """, unsafe_allow_html=True)
                with col_b:
                    st.markdown(f"""
                        <div class='metric-card'>
                            <p style='color: #6b7280; margin: 0; font-size: 12px;'>CLASS</p>
                            <h3 style='color: #000000; margin: 8px 0 0 0;'>{sentiment}</h3>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Interpretation
                st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
                if confidence > 80:
                    st.markdown("""
                        <div style='background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%); 
                        padding: 16px; border-radius: 6px; border-left: 3px solid #4ade80;'>
                            <strong style='color: #000000;'>High Confidence</strong>
                            <p style='margin: 8px 0 0 0; color: #000000;'>
                            The model is highly confident in this prediction.
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                elif confidence > 60:
                    st.markdown("""
                        <div style='background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%); 
                        padding: 16px; border-radius: 6px; border-left: 3px solid #60a5fa;'>
                            <strong style='color: #000000;'>Moderate Confidence</strong>
                            <p style='margin: 8px 0 0 0; color: #000000;'>
                            The model shows reasonable confidence in this prediction.
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                        <div style='background: linear-gradient(135deg, #fef9c3 0%, #fef08a 100%); 
                        padding: 16px; border-radius: 6px; border-left: 3px solid #facc15;'>
                            <strong style='color: #000000;'>Low Confidence</strong>
                            <p style='margin: 8px 0 0 0; color: #000000;'>
                            The prediction may be uncertain. Consider reviewing the text.
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
    
    elif analyze_button:
        st.warning("Please enter text to analyze before proceeding.")
    
    # Minimal Professional Footer
    st.markdown("""
        <div class='footer'>
            <p style='margin: 0; font-size: 14px; color: #94a3b8;'>
                © 2025 Sentiment Analysis System
            </p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
