"""Classify headlines with the trained model.

Usage:
    python src/predict.py "Area man heroically finishes entire meeting without saying anything"
    python src/predict.py            # interactive mode
"""
import os
import sys

import joblib

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import clean_text  # noqa: E402

MODEL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "models", "sarcasm_model.joblib")


def predict(model, headline: str):
    p = model.predict_proba([clean_text(headline)])[0][1]
    return ("Sarcastic" if p >= 0.5 else "Not sarcastic"), p


if __name__ == "__main__":
    if not os.path.exists(MODEL):
        sys.exit("Model not found. Run: python src/train.py")
    m = joblib.load(MODEL)
    if len(sys.argv) > 1:
        label, p = predict(m, " ".join(sys.argv[1:]))
        print(f"{label} (sarcasm probability = {p:.2f})")
    else:
        while True:
            h = input("Headline (blank to quit): ").strip()
            if not h:
                break
            label, p = predict(m, h)
            print(f"  -> {label} (p = {p:.2f})")
