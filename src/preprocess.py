import re
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")
STOPWORDS = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)        # remove mentions
    text = re.sub(r"#\w+", "", text)        # remove hashtags
    text = re.sub(r"[^a-z\s]", "", text)    # remove special chars
    text = re.sub(r"\s+", " ", text).strip()

    # remove stopwords
    text = " ".join([w for w in text.split() if w not in STOPWORDS])
    return text
