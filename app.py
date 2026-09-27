import streamlit as st
import joblib
import re
import os
import pandas as pd


# -------------------------------------------------------------
# Page Configuration
# -------------------------------------------------------------
st.set_page_config(
    page_title="AI Sentiment Analysis System",
    page_icon="💬",
    layout="centered"
)

# -------------------------------------------------------------
# Custom Styling
# -------------------------------------------------------------
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #6c757d;
        margin-bottom: 1.5rem;
    }
    .sentiment-badge {
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
    }
    .badge-positive {
        background-color: #d1e7dd;
        color: #0f5132;
        border: 1px solid #badbcc;
    }
    .badge-negative {
        background-color: #f8d7da;
        color: #842029;
        border: 1px solid #f5c2c7;
    }
    .badge-neutral {
        background-color: #cff4fc;
        color: #055160;
        border: 1px solid #b6effb;
    }
    .badge-text {
        font-size: 1.8rem;
        font-weight: 700;
    }
    .prob-metric {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 0.8rem;
        text-align: center;
        border: 1px solid #e9ecef;
    }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------
# Text Preprocessing
# -------------------------------------------------------------
def clean_text(text: str) -> str:
    """Preprocesses raw review text by lowercasing, stripping URLs,
    removing non-alphabetic characters, and normalizing whitespace."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -------------------------------------------------------------
# Load Trained Model Pipeline
# -------------------------------------------------------------
@st.cache_resource
def load_sentiment_pipeline():
    model_path = os.path.join(os.path.dirname(__file__), "models", "sentiment_pipeline.pkl")
    if not os.path.exists(model_path):
        st.error(f"Trained model not found at {model_path}. Please run the training pipeline first.")
        st.stop()
    return joblib.load(model_path)


pipeline = load_sentiment_pipeline()




# -------------------------------------------------------------
# Application Header
# -------------------------------------------------------------
st.markdown('<div class="main-title">💬 AI-Powered Sentiment Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Analyze customer feedback, product reviews, and survey responses with machine learning.</div>', unsafe_allow_html=True)


# -------------------------------------------------------------
# Quick Sample Test Reviews
# -------------------------------------------------------------
st.subheader("Test Examples")
sample_reviews = [
    "Select an example to auto-fill...",
    "This product is terrible.",
    "I hate this product.",
    "This is the worst product ever.",
    "Very bad quality and completely disappointing.",
    "I am extremely unhappy with this purchase.",
    "The product is broken and completely useless.",
    "I absolutely loved this product.",
    "This product has excellent quality.",
    "The quality is okay, nothing special."
]

selected_sample = st.selectbox(
    "Choose a sample test review to evaluate the model:",
    sample_reviews,
    index=0
)

# Manage text input state
default_text = ""
if selected_sample != "Select an example to auto-fill...":
    default_text = selected_sample

review_input = st.text_area(
    "Enter your review sentence/text:",
    value=default_text,
    placeholder="Type or paste your customer review here (e.g., 'Very bad quality and completely disappointing')...",
    height=110
)

col_btn, col_clear = st.columns([1, 4])
with col_btn:
    analyze_clicked = st.button("🚀 Analyze Sentiment", type="primary", use_container_width=True)


# -------------------------------------------------------------
# Sentiment Prediction & Probabilities
# -------------------------------------------------------------
if analyze_clicked or (selected_sample != "Select an example to auto-fill..." and review_input.strip() != ""):
    if review_input.strip() == "":
        st.warning("⚠️ Please enter a review text to analyze.")
    else:
        cleaned = clean_text(review_input)

        if not cleaned:
            st.warning("⚠️ The review contains no valid alphabetic text after cleaning. Please enter a meaningful sentence.")
        else:
            # Generate Prediction
            predicted_class = pipeline.predict([cleaned])[0]
            classes = list(pipeline.classes_)

            # Probability Breakdown
            if hasattr(pipeline, "predict_proba"):
                probabilities = pipeline.predict_proba([cleaned])[0]
                prob_dict = {cls_name: float(prob) for cls_name, prob in zip(classes, probabilities)}
                confidence = prob_dict[predicted_class]
            else:
                prob_dict = None
                confidence = 1.0

            st.divider()
            st.subheader("📊 Analysis Results")

            # Sentiment Banner
            if predicted_class == "Positive":
                st.markdown(f"""
                <div class="sentiment-badge badge-positive">
                    <div style="font-size: 1rem; text-transform: uppercase; letter-spacing: 1px;">Predicted Sentiment</div>
                    <div class="badge-text">😊 Positive</div>
                    <div style="margin-top: 0.5rem; font-size: 1.1rem; font-weight: 500;">Confidence: {confidence * 100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

            elif predicted_class == "Negative":
                st.markdown(f"""
                <div class="sentiment-badge badge-negative">
                    <div style="font-size: 1rem; text-transform: uppercase; letter-spacing: 1px;">Predicted Sentiment</div>
                    <div class="badge-text">😞 Negative</div>
                    <div style="margin-top: 0.5rem; font-size: 1.1rem; font-weight: 500;">Confidence: {confidence * 100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

            else:
                st.markdown(f"""
                <div class="sentiment-badge badge-neutral">
                    <div style="font-size: 1rem; text-transform: uppercase; letter-spacing: 1px;">Predicted Sentiment</div>
                    <div class="badge-text">😐 Neutral</div>
                    <div style="margin-top: 0.5rem; font-size: 1.1rem; font-weight: 500;">Confidence: {confidence * 100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

            # Confidence / Probability Score Distribution
            if prob_dict:
                st.write("#### Confidence / Class Probabilities")
                col_neg, col_neu, col_pos = st.columns(3)

                neg_prob = prob_dict.get("Negative", 0.0)
                neu_prob = prob_dict.get("Neutral", 0.0)
                pos_prob = prob_dict.get("Positive", 0.0)

                with col_neg:
                    st.metric(label="😞 Negative", value=f"{neg_prob * 100:.1f}%")
                    st.progress(neg_prob)

                with col_neu:
                    st.metric(label="😐 Neutral", value=f"{neu_prob * 100:.1f}%")
                    st.progress(neu_prob)

                with col_pos:
                    st.metric(label="😊 Positive", value=f"{pos_prob * 100:.1f}%")
                    st.progress(pos_prob)
