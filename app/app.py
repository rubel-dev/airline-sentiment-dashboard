import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from wordcloud import WordCloud
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

 
 
st.set_page_config(
    page_title="Airline Sentiment SaaS Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

 
 
st.markdown("""
<style>
    .main-title {
        font-size: 38px;
        font-weight: 800;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 16px;
        color: #6b7280;
        margin-top: 0px;
    }
    .metric-card {
        padding: 18px;
        border-radius: 14px;
        background: white;
        border: 1px solid #e5e7eb;
        box-shadow: 0px 2px 8px rgba(0,0,0,0.03);
        text-align: center;
    }
    .metric-label {
        font-size: 14px;
        color: #6b7280;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 26px;
        font-weight: 800;
        color: #111827;
    }
    .section-header {
        font-size: 20px;
        font-weight: 700;
        margin-top: 10px;
        margin-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)

 
# Load Model & Dataset
 
model = joblib.load("models/sentiment_model.pkl")
df = pd.read_csv("data/Tweets.csv")

# Fix missing values
df["text"] = df["text"].fillna("")
df["airline_sentiment"] = df["airline_sentiment"].fillna("unknown")
df["airline"] = df["airline"].fillna("unknown")
df["negativereason"] = df["negativereason"].fillna("unknown")

 
# Header
 
st.markdown('<div class="main-title"> Airline Tweet Sentiment Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">NLP + ML dashboard for customer sentiment insights and real-time prediction</div>', unsafe_allow_html=True)
st.markdown("---")


# Sidebar Filters

st.sidebar.header(" Filters")

selected_airline = st.sidebar.selectbox(
    "Select Airline",
    ["All"] + sorted(df["airline"].unique().tolist())
)

selected_sentiment = st.sidebar.selectbox(
    "Select Sentiment",
    ["All", "positive", "neutral", "negative"]
)

# Filter Dataset
filtered_df = df.copy()
if selected_airline != "All":
    filtered_df = filtered_df[filtered_df["airline"] == selected_airline]

if selected_sentiment != "All":
    filtered_df = filtered_df[filtered_df["airline_sentiment"] == selected_sentiment]

st.sidebar.markdown("---")
st.sidebar.write(f" Total Tweets: **{len(filtered_df)}**")

 
 
pos_count = (filtered_df["airline_sentiment"] == "positive").sum()
neu_count = (filtered_df["airline_sentiment"] == "neutral").sum()
neg_count = (filtered_df["airline_sentiment"] == "negative").sum()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Tweets</div>
        <div class="metric-value">{len(filtered_df)}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Positive</div>
        <div class="metric-value">{pos_count}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Neutral</div>
        <div class="metric-value">{neu_count}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Negative</div>
        <div class="metric-value">{neg_count}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("")

 
# Download Filtered Dataset
 
csv_data = filtered_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label=" Download Filtered Dataset (CSV)",
    data=csv_data,
    file_name="filtered_airline_tweets.csv",
    mime="text/csv"
)

 
# Tabs Layout
 
tab1, tab2, tab3, tab4 = st.tabs([" Analytics", " Predict Sentiment", " Bigram Trends", " WordCloud"])

 
# TAB 1: Analytics
 
with tab1:
    st.markdown('<div class="section-header">Sentiment Distribution</div>', unsafe_allow_html=True)

    sentiment_counts = filtered_df["airline_sentiment"].value_counts()

    fig, ax = plt.subplots()
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, ax=ax)
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    st.markdown('<div class="section-header">Top Negative Reasons</div>', unsafe_allow_html=True)
    neg_reason_counts = filtered_df[filtered_df["airline_sentiment"] == "negative"]["negativereason"].value_counts().head(10)

    if len(neg_reason_counts) > 0:
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        sns.barplot(x=neg_reason_counts.values, y=neg_reason_counts.index, ax=ax2)
        ax2.set_xlabel("Count")
        ax2.set_ylabel("Reason")
        st.pyplot(fig2)
    else:
        st.warning("No negative reasons available for the selected filter.")

 
# Confidence Gauge
 
def plot_confidence_gauge(conf):
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.axis("off")

    ax.barh(0, 1, height=0.35)
    ax.barh(0, conf, height=0.35)

    ax.text(0, 0.45, "0%", fontsize=12)
    ax.text(0.95, 0.45, "100%", fontsize=12)
    ax.text(conf, -0.2, f"{conf*100:.1f}%", fontsize=14, fontweight="bold")

    ax.set_xlim(0, 1)
    return fig

 
# TAB 2: Predict Sentiment
 
with tab2:
    st.markdown('<div class="section-header">Real-Time Tweet Sentiment Prediction</div>', unsafe_allow_html=True)
    user_tweet = st.text_area("Enter Tweet Text")

    if st.button(" Predict Sentiment"):
        if user_tweet.strip() == "":
            st.warning("Please enter a tweet.")
        else:
            pred = model.predict([user_tweet])[0]
            prob = model.predict_proba([user_tweet])[0]
            confidence = prob.max()

            st.success(f" Predicted Sentiment: **{pred.upper()}**")
            st.info(f" Confidence Score: **{confidence:.2f}**")

            st.markdown('<div class="section-header">Confidence Gauge</div>', unsafe_allow_html=True)
            fig3 = plot_confidence_gauge(confidence)
            st.pyplot(fig3)

            st.markdown('<div class="section-header">Probability Breakdown</div>', unsafe_allow_html=True)
            proba_df = pd.DataFrame({
                "Sentiment": model.classes_,
                "Probability": prob
            })
            st.dataframe(proba_df, use_container_width=True)

 
# TAB 3: Bigram Trends (Keyword Trends)
 
with tab3:
    st.markdown('<div class="section-header">Trending Complaint Bigrams (Negative Tweets)</div>', unsafe_allow_html=True)

    negative_texts = filtered_df[filtered_df["airline_sentiment"] == "negative"]["text"].tolist()

    if len(negative_texts) > 0:
        vectorizer = CountVectorizer(stop_words="english", ngram_range=(2, 2), max_features=20)
        X = vectorizer.fit_transform(negative_texts)

        bigram_counts = X.sum(axis=0)
        bigrams = vectorizer.get_feature_names_out()
        counts = np.array(bigram_counts).flatten()

        bigram_df = pd.DataFrame({"Bigram": bigrams, "Count": counts}).sort_values(by="Count", ascending=False)

        fig4, ax4 = plt.subplots(figsize=(10, 4))
        sns.barplot(x="Count", y="Bigram", data=bigram_df, ax=ax4)
        ax4.set_xlabel("Count")
        ax4.set_ylabel("Bigram")
        st.pyplot(fig4)

    else:
        st.warning("No negative tweets available for this filter, so bigram trends cannot be generated.")

 
# TAB 4: WordCloud
 
with tab4:
    st.markdown('<div class="section-header">WordCloud for Negative Tweets</div>', unsafe_allow_html=True)

    neg_text = " ".join(filtered_df[filtered_df["airline_sentiment"] == "negative"]["text"].dropna().astype(str).tolist())

    if neg_text.strip():
        wc = WordCloud(width=900, height=400, background_color="white").generate(neg_text)

        fig5, ax5 = plt.subplots(figsize=(12, 6))
        ax5.imshow(wc, interpolation="bilinear")
        ax5.axis("off")
        st.pyplot(fig5)
    else:
        st.warning("No negative tweets available for the selected filters.")
