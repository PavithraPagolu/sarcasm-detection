"""Feature engineering: simple hand-crafted style features (used for EDA / reporting)."""
import pandas as pd


def add_style_features(df: pd.DataFrame, col: str = "headline") -> pd.DataFrame:
    """Stylistic cues that often correlate with sarcasm."""
    s = df[col].astype(str)
    df = df.copy()
    df["n_words"] = s.str.split().str.len()
    df["n_chars"] = s.str.len()
    df["has_quote"] = s.str.contains(r"['\"]").astype(int)
    df["has_exclaim_or_q"] = s.str.contains(r"[!?]").astype(int)
    df["n_upper_words"] = s.apply(lambda x: sum(w.isupper() and len(w) > 1 for w in x.split()))
    return df
