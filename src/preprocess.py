"""Text preprocessing: lowercase, strip noise, remove stopwords, lemmatize (NLTK)."""
import re

import nltk


# --- NLTK resources (download once; fall back gracefully if offline) -------------
def _ensure(resource, name):
    try:
        nltk.data.find(resource)
        return True
    except LookupError:
        try:
            nltk.download(name, quiet=True)
            nltk.data.find(resource)
            return True
        except Exception:
            return False


_HAS_STOP = _ensure("corpora/stopwords", "stopwords")
_HAS_WN = _ensure("corpora/wordnet", "wordnet")

if _HAS_STOP:
    from nltk.corpus import stopwords
    STOP_WORDS = set(stopwords.words("english"))
else:  # offline fallback
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    STOP_WORDS = set(ENGLISH_STOP_WORDS)

# Keep negations - they often carry sarcastic tone
STOP_WORDS -= {"no", "not", "nor", "never"}

if _HAS_WN:
    from nltk.stem import WordNetLemmatizer
    _norm = WordNetLemmatizer().lemmatize
else:  # offline fallback
    from nltk.stem import PorterStemmer
    _norm = PorterStemmer().stem


def clean_text(text: str) -> str:
    """Return a cleaned, lemmatized version of a headline."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)   # urls
    text = re.sub(r"[^a-z\s']", " ", text)          # keep letters only
    tokens = [t for t in text.split() if t not in STOP_WORDS and len(t) > 1]
    return " ".join(_norm(t) for t in tokens)
