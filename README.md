# Airline Tweet Sentiment Dashboard
🔗 Project Links

Live: [click me](https://airline-sentiment-dashboard-k9.streamlit.app/)

##  Overview
This project is an NLP-based web application that classifies airline-related tweets into
Positive, Neutral, or Negative sentiment. It is designed to help analyze customer feedback
and social media sentiment in real time.

The system follows a clean end-to-end pipeline:
text preprocessing → feature extraction → model training → deployment.

---

## Problem Statement
Airlines receive a large volume of customer feedback through Twitter. Manually analyzing
these tweets is time-consuming and inefficient. Automating sentiment classification helps
identify negative trends, improve response time, and enhance customer experience.

---

##  Approach

### 1. Text Preprocessing
- Converted text to lowercase
- Removed URLs, mentions, punctuation, and extra whitespace
- Cleaned tweets to reduce noise before modeling

### 2. Feature Engineering
- Used TF-IDF vectorization to transform text into numerical features
- Captured important words that influence sentiment classification

### 3. Modeling
- Trained a Logistic Regression / Linear SVM classifier
- Chosen for efficiency, interpretability, and strong performance on sparse TF-IDF features

### 4. Evaluation
- Evaluated using accuracy and confusion matrix
- Achieved ~0.91 accuracy on validation data

---

##  Web Application
- Built with Streamlit
- User can enter any tweet and receive:
  - Predicted sentiment
  - Confidence score
- Simple and intuitive interface suitable for non-technical users

---

##  Results
- Sentiment Accuracy: 0.91
- Fast inference suitable for real-time usage

---

##  How to Run Locally

### Install dependencies
```bash
pip install -r requirements.txt
Run the app
bash
Copy code
streamlit run streamlit_app.py


🛠 Tech Stack
Python

Pandas, NumPy

Scikit-learn

TF-IDF (NLP)

Streamlit



 
 


