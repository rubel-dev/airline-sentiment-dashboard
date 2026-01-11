Airline Tweet Sentiment Dashboard
 Overview

This project classifies tweets into Positive, Neutral, or Negative sentiment to enable quick analysis of customer feedback. It provides a browser-based dashboard with real-time sentiment predictions and analytics.

 Built with Python, Scikit-Learn, NLP preprocessing, and Streamlit.

 Problem

Airlines receive thousands of tweets daily. Automating sentiment analysis helps customer experience teams quickly identify negative trends and patterns.

 Approach
🔹 Preprocessing

Lowercase conversion

Removal of noise (URLs, punctuation, stopwords)

TF-IDF vectorization
 Modeling

Trained Logistic Regression and Linear SVM classifiers

Evaluated with accuracy and confusion matrix

Produced sentiment tags with confidence levels

 Dashboard

Input form for live tweet classification

Displays sentiment probabilities and top insights

 Metrics

Accuracy: 0.91

Consistent performance on test split

 Run Locally
Install
pip install -r requirements.txt

Run
streamlit run streamlit_app.py

🔗 Demo & Source

Live: [click me](https://airline-sentiment-dashboard-k9.streamlit.app/)
 

🛠 Tech Stack

Python | Scikit-Learn | TF-IDF | NLP | Streamlit
