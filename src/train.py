import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from preprocess import clean_text

def train_model():
    df = pd.read_csv("data/Tweets.csv")

    
    df = df[["text", "airline_sentiment", "airline", "negativereason"]]

   
    df["clean_text"] = df["text"].apply(clean_text)

    X = df["clean_text"]
    y = df["airline_sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=7000, ngram_range=(1,2))),
        ("clf", LogisticRegression(max_iter=200))
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(" Accuracy:", accuracy_score(y_test, y_pred))
    print("\n Classification Report:\n", classification_report(y_test, y_pred))

    joblib.dump(model, "models/sentiment_model.pkl")
    print(" Model saved to models/sentiment_model.pkl")

if __name__ == "__main__":
    train_model()
