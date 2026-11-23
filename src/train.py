"""Train + evaluate TF-IDF + Logistic Regression sarcasm classifier.

Usage:
    python src/train.py                      # auto-detects dataset
    python src/train.py --data data/Sarcasm_Headlines_Dataset_v2.json
"""
import argparse
import glob
import os
import sys
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(__file__))
from features import add_style_features  # noqa: E402
from preprocess import clean_text  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_data(path=None) -> pd.DataFrame:
    if path is None:
        found = sorted(glob.glob(os.path.join(ROOT, "data", "Sarcasm_Headlines_Dataset*.json")))
        path = found[-1] if found else os.path.join(ROOT, "data", "sample_headlines.csv")
    print(f"[data] loading {path}")
    if path.endswith(".json"):
        df = pd.read_json(path, lines=True)       # Kaggle dataset is JSON-lines
    else:
        df = pd.read_csv(path)
    df = df[["headline", "is_sarcastic"]].dropna().drop_duplicates("headline")
    print(f"[data] {len(df)} headlines | sarcastic ratio = {df.is_sarcastic.mean():.2f}")
    return df


def main(args):
    t0 = time.time()
    df = load_data(args.data)
    df = add_style_features(df)
    df["clean"] = df["headline"].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["is_sarcastic"], test_size=0.2, random_state=42,
        stratify=df["is_sarcastic"])

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(sublinear_tf=True, min_df=2 if len(df) > 1000 else 1)),
        ("clf", LogisticRegression(max_iter=1000, solver="liblinear")),
    ])
    grid = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "clf__C": [0.5, 1, 4],
    }
    cv = 5 if len(df) > 500 else 3
    search = GridSearchCV(pipe, grid, cv=cv, scoring="f1", n_jobs=-1)
    search.fit(X_train, y_train)
    model = search.best_estimator_
    print(f"[train] best params: {search.best_params_}")

    pred = model.predict(X_test)
    print(f"\nAccuracy : {accuracy_score(y_test, pred):.4f}")
    print(f"F1-score : {f1_score(y_test, pred):.4f}\n")
    print(classification_report(y_test, pred, target_names=["Not Sarcastic", "Sarcastic"]))

    # Top words driving each class
    vec, clf = model.named_steps["tfidf"], model.named_steps["clf"]
    names = vec.get_feature_names_out()
    order = clf.coef_[0].argsort()
    print("Top 'sarcastic' terms :", ", ".join(names[order[-10:]][::-1]))
    print("Top 'normal' terms    :", ", ".join(names[order[:10]]))

    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
    joblib.dump(model, os.path.join(ROOT, "models", "sarcasm_model.joblib"))
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, pred, display_labels=["Not Sarc.", "Sarcastic"], cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "outputs", "confusion_matrix.png"), dpi=150)
    print(f"\n[done] model saved to models/sarcasm_model.joblib in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="path to .json (Kaggle) or .csv dataset")
    main(ap.parse_args())
