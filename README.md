# 💬 AI-Powered Sentiment Analysis System

An end-to-end Natural Language Processing (NLP) and Machine Learning system that classifies customer reviews into **Positive**, **Negative**, or **Neutral** sentiment with calibrated probabilistic confidence scores. Built with scikit-learn, TF-IDF representations, and deployed with an interactive **Streamlit** web application.

---

## 📌 Project Overview

Customer reviews provide critical feedback across products and services, but manual triage is slow, resource-intensive, and subjective. This project automates the sentiment classification workflow:
- **Text Normalization & Preprocessing:** Cleans and normalizes unstructured feedback while preserving critical negations (*"not good"*, *"never buy again"*).
- **Feature Extraction:** Transforms review text into informative numerical vectors via TF-IDF with unigrams and bigrams `(1, 2)` and sublinear term frequency scaling.
- **Calibrated Multi-Class Classification:** Uses regularized Logistic Regression ($C=2.5$) with softmax probability calibration to predict class labels and confidence distributions.
- **Interactive Web Interface:** Provides real-time review evaluation, preset test benchmarks, and visual confidence gauges via Streamlit.

---

## 📂 Project Directory Structure

```text
sentiment-analysis/
├── data/
│   └── reviews.csv                  # 10,092 customer reviews with sentiment labels & metadata
├── notebooks/
│   └── sentiment_analysis.ipynb     # Complete Jupyter Notebook (EDA, preprocessing, modeling, tuning)
├── models/
│   └── sentiment_pipeline.pkl       # Serialized scikit-learn pipeline (TF-IDF + Logistic Regression)
├── images/
│   ├── demo.mp4                     # High-definition video demonstration of the application
│   ├── demo.webp                    # Animated demonstration preview
│   ├── sentiment_distribution.png   # Class distribution plot
│   ├── confusion_matrix.png         # Final evaluation confusion matrix
│   ├── roc_auc_curve.png            # Multi-class ROC & AUC curve
│   └── word_count_distribution.png  # Review word count distribution
├── app.py                           # Interactive Streamlit web application
├── requirements.txt                 # Project dependencies
├── EVALUATION_REPORT.md             # In-depth model comparison & evaluation report
└── README.md                        # Project overview, setup, and evaluation guide
```

---

## 🎥 Application Demonstration

Below is a demonstration preview of the live Streamlit application analyzing positive, negative, and neutral customer feedback:

![Application Demo](images/demo.webp)

> 📹 **Full Video:** You can also watch the high-definition MP4 recording at [images/demo.mp4](images/demo.mp4).

---

## 🚀 Setup Guide

### 1. Prerequisites
- Python 3.9+
- Recommended: Virtual environment (`venv` or `conda`)

### 2. Environment Setup & Dependency Installation
Clone or navigate to the project directory and create a virtual environment:

```powershell
# Navigate to project directory
cd d:\sentiment-analysis

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install required dependencies
pip install -r requirements.txt
```

### 3. Run the Interactive Web Application
Launch the Streamlit web dashboard:

```powershell
streamlit run app.py
```

The application will start locally and open in your default browser at:
`http://localhost:8501`

---

## 📊 Evaluation Guide

### 1. In-Depth Evaluation Report
For the complete empirical comparison of models (Multinomial Naive Bayes, Linear SVC, Logistic Regression), confusion matrices, 5-fold cross-validation scores, and selection justification, refer to:
👉 **[EVALUATION_REPORT.md](EVALUATION_REPORT.md)**

### 2. Reproducing Evaluation via Jupyter Notebook
Launch the Jupyter Notebook to re-run the exploratory data analysis, feature ablation experiments, cross-validation, and pipeline serialization:

```powershell
jupyter notebook notebooks/sentiment_analysis.ipynb
```

### 3. Quick Command-Line Evaluation & Sanity Check
You can evaluate the serialized pipeline on benchmark review sentences directly from the command line:

```powershell
python -c "
import joblib, re

pipeline = joblib.load('models/sentiment_pipeline.pkl')
clean = lambda t: re.sub(r'\s+', ' ', re.sub(r'[^a-zA-Z\s]', '', str(t).lower())).strip()

test_cases = [
    'This product has excellent quality.',
    'The product is broken and completely useless.',
    'The quality is okay, nothing special.'
]

for review in test_cases:
    c = clean(review)
    pred = pipeline.predict([c])[0]
    probs = dict(zip(pipeline.classes_, [round(float(p)*100, 1) for p in pipeline.predict_proba([c])[0]]))
    print(f'Review: \"{review}\"')
    print(f'Prediction: {pred} | Probabilities: {probs}\n')
"
```
