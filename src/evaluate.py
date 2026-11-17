"""Evaluate zero-shot pretrained FinBERT on the PhraseBank test split.

Reads data/processed/phrasebank_test_scored.csv (true `label` + FinBERT
`predicted_label`) and writes results/metrics_phrasebank.json with accuracy,
per-class precision/recall/F1 and the confusion matrix, plus a majority-class
baseline for comparison. Prints everything; nothing here is estimated.
"""
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["positive", "neutral", "negative"]


def main() -> None:
    df = pd.read_csv(ROOT / "data" / "processed" / "phrasebank_test_scored.csv")
    y_true, y_pred = df["label"], df["predicted_label"]
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, labels=LABELS,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    majority = y_true.value_counts().idxmax()
    majority_acc = (y_true == majority).mean()

    metrics = {
        "n_test": int(len(df)),
        "model": "ProsusAI/finbert (pretrained, zero-shot, no fine-tuning)",
        "accuracy": round(float(acc), 4),
        "majority_class": majority,
        "majority_baseline_accuracy": round(float(majority_acc), 4),
        "per_class": {lab: {k: round(float(v), 4) for k, v in report[lab].items()
                            if k in ("precision", "recall", "f1-score", "support")}
                      for lab in LABELS},
        "confusion_matrix_labels": LABELS,
        "confusion_matrix_rows_true_cols_pred": [[int(x) for x in row] for row in cm],
    }
    out = ROOT / "results" / "metrics_phrasebank.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
