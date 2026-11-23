# &#x20;Sarcasm Detection in News Headlines

An NLP classification project that detects whether a news headline is **sarcastic** or **not sarcastic**, using **TF-IDF** features and **Logistic Regression**.

**Tech stack:** Python · Scikit-learn · NLTK · Pandas · Matplotlib

## Highlights

* Text preprocessing with NLTK (lowercasing, noise removal, stop-word removal, lemmatization; negations like *not/never* are kept since they carry tone)
* Feature engineering: TF-IDF (unigrams + bigrams, sublinear TF) and stylistic features (length, quotes, punctuation, caps)
* `GridSearchCV` hyperparameter tuning (n-gram range, regularization `C`) optimizing F1
* Evaluation: accuracy, precision, recall, F1, confusion matrix, top predictive terms
* Trains in **seconds** on CPU — no GPU needed

## Project structure

```
sarcasm-detection/
├── data/
│   └── sample\_headlines.csv     # tiny demo set so the project runs instantly
├── src/
│   ├── preprocess.py            # NLTK text cleaning
│   ├── features.py              # stylistic features
│   ├── train.py                 # train + tune + evaluate + save model
│   └── predict.py               # classify new headlines
├── models/                      # saved model (generated)
├── outputs/                     # confusion matrix (generated)
└── requirements.txt
```

## Dataset

Uses the **News Headlines Dataset for Sarcasm Detection** (Rishabh Misra, Kaggle) — \~28k headlines from *The Onion* (sarcastic) and *HuffPost* (real).

1. Download from Kaggle: search "News Headlines Dataset For Sarcasm Detection".
2. Put `Sarcasm\_Headlines\_Dataset\_v2.json` (or `Sarcasm\_Headlines\_Dataset.json`) inside `data/`.

If no dataset is found, the code automatically falls back to `data/sample\_headlines.csv` (50 headlines, for a quick smoke test only — metrics on it are not meaningful).

## Quick start

```bash
git clone https://github.com/<your-username>/sarcasm-detection.git
cd sarcasm-detection
python -m venv venv \&\& source venv/bin/activate   # Windows: venv\\Scripts\\activate
pip install -r requirements.txt

python src/train.py                                # trains \& evaluates
python src/predict.py "Area man thrilled to attend another meeting that could have been an email"
```

NLTK data (stopwords, wordnet) downloads automatically on first run.

## Results

Run `python src/train.py` on the full dataset and paste your numbers here, e.g.:

|Metric|Score|
|-|-|
|Accuracy|*80.3%*|
|F1 (sarcastic)|*0.77*|

TF-IDF + Logistic Regression typically reaches roughly 79–80% accuracy on this dataset.

!\[Confusion matrix](outputs/confusion\_matrix.png)

## Limitations \& future work

* Bag-of-words ignores word order and context; sarcasm sometimes needs world knowledge.
* Source bias: *The Onion* vs. *HuffPost* style may be learned instead of pure sarcasm.
* Next: try LinearSVC / Naive Bayes baselines, word embeddings, or a fine-tuned BERT model.

## License

MIT

